import pyaudio
import numpy
import math
import time
from functools import reduce

def numpyMapOneDimensional( func, values ):
	output = numpy.empty_like( values )
	for i in range( values.size ):
		output[ i ] = func( values[ i ] )
	return output
	
def scale( x, xMin, xMax ):
	return min( 1, max( 0, ( x - xMin ) ) / ( xMax - xMin ) )

class Int16Settings:
	def __init__( self ):
		self.npArrayDtype = numpy.int16
		self.npSumDtype = numpy.uint64
		self.npDifferenceDtype = numpy.int64
		self.paFormat = pyaudio.paInt16
		self.maxValue = 1 << 15		

class Float32Settings:
	def __init__( self ):
		self.npArrayDtype = numpy.float32
		self.npSumDtype = numpy.float32
		self.npDifferenceDtype = numpy.float32
		self.paFormat = pyaudio.paFloat32
		self.maxValue = numpy.float32( 1 )		

class RollingPowerWindowEvaluator:
	def __init__( self, audioBufferLength, smoothingWindowHalfSize, typeSettings ):
		self.smoothingWindowHalfSize = smoothingWindowHalfSize
		self.typeSettings = typeSettings

		self.totalPower = numpy.float32( 0 )
		self.totalSquaresOfPower = numpy.float32( 0 )
		self.powerStdDevEstimateTimesBufferSize = numpy.float32( 0 )
		self.outlyingPowerThreshhold = numpy.float32( 0 )
		self.totalOutlyingPower = numpy.float32( 0 )
		
		self.amplitudeBuffer = numpy.zeros( audioBufferLength, dtype = self.typeSettings.npArrayDtype )
		self.powerWindowBuffer = numpy.zeros( audioBufferLength, dtype = numpy.float32 )
		self.outlyingPowerBuffer = numpy.zeros( audioBufferLength, dtype = numpy.float32 )

		# make odd length smoothing window so it can be centered on an integer index
		smoothingWindowSize = ( self.smoothingWindowHalfSize * 2 ) + 1
		# simple average of neigbors... each weighted 1 / window size
		self.smoothingWindowFunction = numpy.full( smoothingWindowSize, 1 / float( smoothingWindowSize ) )
		
	def addData( self, data ):
		newAmplitudes = numpy.abs( data )

		self.amplitudeBuffer = numpy.roll( self.amplitudeBuffer, -newAmplitudes.size )
		self.amplitudeBuffer[ -newAmplitudes.size: ] = newAmplitudes
		
		self.powerWindowBuffer = numpy.roll( self.powerWindowBuffer, -newAmplitudes.size )
		
		# Convolve applies the smoothing function to the amplitudes.
		# Boundary effects exist for output[:halfWindowSize] and output[-halfWindowSize:]
		# Boundary effects from the last update have been rolled down the buffer to powerWindowBuffer[ -(newAmplitudes.size + halfWindowSize):-newAmplitudes.size ]
		# To correct those (now that there are enough values), convlove must run on input values starting an additional halfWindowSize earlier ( -(newAmplitudes.size + halfWindowSize + halfWindowSize) ), discarding the front-end boundary errors: output[:halfWindowSize]
		replacementIndex = -( newAmplitudes.size + self.smoothingWindowHalfSize )
		outgoingPowers = self.powerWindowBuffer[ replacementIndex: ]
		self.totalPower -= outgoingPowers.sum()
		self.totalSquaresOfPower -= numpy.dot( outgoingPowers, outgoingPowers )
		incomingPowers = numpy.convolve(
			self.amplitudeBuffer[ ( replacementIndex - self.smoothingWindowHalfSize - 5): ],
			self.smoothingWindowFunction,
			'same' )[ replacementIndex: ]
		self.powerWindowBuffer[ replacementIndex: ] = incomingPowers
			
		self.totalPower += incomingPowers.sum()
		self.totalSquaresOfPower += numpy.dot( incomingPowers, incomingPowers )
		self.powerStdDevEstimateTimesBufferSize = math.sqrt( max( 0, ( self.powerWindowBuffer.size * self.totalSquaresOfPower ) - ( self.totalPower * self.totalPower ) ) )
		self.outlyingPowerThreshhold = ( self.totalPower + self.powerStdDevEstimateTimesBufferSize ) / self.powerWindowBuffer.size

		self.outlyingPowerBuffer = numpy.roll( self.outlyingPowerBuffer, -newAmplitudes.size )		
		self.totalOutlyingPower -= self.outlyingPowerBuffer[ replacementIndex: ].sum()
		self.outlyingPowerBuffer[ replacementIndex: ] = numpy.clip( self.powerWindowBuffer[ replacementIndex: ] - self.outlyingPowerThreshhold, 0, self.typeSettings.maxValue )
		self.totalOutlyingPower += self.outlyingPowerBuffer[ replacementIndex: ].sum()
		
		# unsurprisingly too slow most of the time, calculating power over current overall threshhold for each incremental data instead, which isn't the same thing, but maybe close enough
		#self.totalOutlyingPower = reduce( lambda sum, value: sum + max( 0, value - self.outlyingPowerThreshhold ), self.powerWindowBuffer )	
	
	def getTotalPowerPercent( self ):
		return self.totalPower / ( self.amplitudeBuffer.size * float( self.typeSettings.maxValue ) )
	
	def getOutlyingPowerPercent( self ):
		return self.totalOutlyingPower / ( self.amplitudeBuffer.size * ( self.typeSettings.maxValue - self.outlyingPowerThreshhold ) )
		
	def getPowerCoefficientOfVariation( self ):
		if self.totalPower == 0:
			return 0
		return self.powerStdDevEstimateTimesBufferSize / self.totalPower
		
	def getTalkLevelEstimate( self, cvMin = 0.5, cvMax = 0.9, pMin = 0.005, pMax = 0.02 ):
		return scale( self.getPowerCoefficientOfVariation(), cvMin, cvMax ) * scale( self.getTotalPowerPercent(), pMin, pMax )
	
class AudioMonitor:
	def __init__( self, audioInterface, deviceIndex = 0, bufferSeconds = 3 ):
		self.audioInterface = audioInterface
		self.deviceIndex = deviceIndex
		self.bufferSeconds = bufferSeconds
		self.active = False
		self.muteDetected = False
		self.audioInputStream = None
		self.typeSettings = None
		self.audioInputFrameCount = 4096
		
	def initializeStream( self ):
		deviceInfo = audioInterface.get_device_info_by_index( micDeviceIndex )
		sampleRate = round( deviceInfo[ 'defaultSampleRate' ] / 2 )
		audioBufferLength = math.ceil( sampleRate * self.bufferSeconds )
		
		# half of 85Hz wave, the low end of human voice, since taking abs value, and then half again b/c left and right neighbors
		smoothingWindowHalfSize = round( sampleRate / ( 85 * 2 * 2 ) )

		# TODO interrogate deviceInfo
		self.typeSettings = Int16Settings()
#		self.typeSettings = Float32Settings()

		self.rollingPowerWindowEvaluator = RollingPowerWindowEvaluator( audioBufferLength, smoothingWindowHalfSize, self.typeSettings )
		
		self.audioInputStream = audioInterface.open( 
			input_device_index = self.deviceIndex,
			format = self.typeSettings.paFormat,
			channels = 1,
			rate = sampleRate,
			input = True,
			frames_per_buffer = self.audioInputFrameCount,
			stream_callback = lambda data, frameCount, timeInfo, statusFlags: self.receiveData( data, frameCount, timeInfo, statusFlags ) )
	
	def start( self ):
		if self.audioInputStream is None:
			self.initializeStream()
		else:
			self.audioInputStream.start_stream()
			
		self.active = True
	
	def stop( self ):
		self.active = False
		if self.audioInputStream is None:
			return
			
		# Documentation fuzzy on whether this blocks until all pending calls to callback return.
		self.audioInputStream.stop_stream()
	
	def receiveData( self, data, frameCount, timeInfo, statusFlags ):
		if statusFlags & pyaudio.paInputOverflow == pyaudio.paInputOverflow:
			# TODO: buffer of when input overflow occurred during the current input buffer
			print( 'Input overflow!' )
			
		self.rollingPowerWindowEvaluator.addData( numpy.fromstring( data, self.typeSettings.npArrayDtype ) )
			
		self.muteDetected = self.rollingPowerWindowEvaluator.totalPower == 0
		return ( None, pyaudio.paContinue )

def FirstOrDefault( testFunc, values, defaultValue = None ):
	return next( iter( filter( testFunc, values ) ), defaultValue )
	
def GetPreferredDeviceByName( inputDeviceInfos, preferredDeviceName, deviceNameSearchStrings, fallBackToDefaultDevice = False ):
	if preferredDeviceName is not None:
		deviceNameSearchStrings = list( deviceNameSearchStrings )
		deviceNameSearchStrings.insert( 0, preferredDeviceName )

	for searchString in deviceNameSearchStrings:
		searchString = searchString.lower()
		preferredMicDeviceInfo = FirstOrDefault( lambda i: searchString in i[ 'name' ].lower(), inputDeviceInfos )
		if preferredMicDeviceInfo is not None:
			return preferredMicDeviceInfo

	if fallBackToDefaultDevice:
		return inputDeviceInfos[ 0 ]

	return None
	
##### GUI BELOW #####
import tkinter as Tk

class TalxlatCanvas(Tk.Canvas):
	def __init__( self, parent, micMonitor, speakerMonitor, bgColor, fgColor, muteBgColor, muteFgColor, shapeMargin, **kwargs ):
		Tk.Canvas.__init__( self, parent, **kwargs )

		self.micMonitor = micMonitor
		self.speakerMonitor = speakerMonitor
		self.bgColor = bgColor
		self.fgColor = fgColor
		self.muteBgColor = muteBgColor
		self.muteFgColor = muteFgColor
		self.shapeMargin = 10
		self.shapeThickness = 30
		self.micShape = None
		self.speakerShape = None
		self.muteShapes = None

		self.bind( '<Configure>', self.onResize )

		self.createShapes()

	def createShapes( self ):
		width = self.winfo_width()
		height = self.winfo_height()
		
		if self.micShape is None:
			micPoints = [
				self.shapeMargin, height - self.shapeMargin,
				( width / 2 ), ( height / 2 ) + self.shapeMargin,
				width - self.shapeMargin, height - self.shapeMargin,
				width - self.shapeMargin - self.shapeThickness, height - self.shapeMargin,
				width / 2, ( height / 2 ) + self.shapeMargin + ( 4 * self.shapeThickness ),
				self.shapeMargin + self.shapeThickness, height - self.shapeMargin ]
			self.micShape = self.create_polygon( micPoints, fill = '' )
			
		if self.speakerShape is None:
			speakerPoints = [
				self.shapeMargin, self.shapeMargin,
				( width / 2 ), ( height / 2 ) - self.shapeMargin,
				width - self.shapeMargin, self.shapeMargin,
				width - self.shapeMargin - self.shapeThickness, self.shapeMargin,
				width / 2, ( height / 2 ) - self.shapeMargin - ( 4 * self.shapeThickness ),
				self.shapeMargin + self.shapeThickness, self.shapeMargin ]
			self.speakerShape = self.create_polygon( speakerPoints, fill = '' )
		

		if self.muteShapes is None:
			self.muteShapes = []
			eyeOffset = 0.33
			eyeHeight = int( height * 0.4 )
			eyeThickness = int( self.shapeThickness * 1.5 )
			self.muteShapes.append( self.create_oval(
				( width * ( 0.5 - eyeOffset ) ) - eyeThickness, eyeHeight - ( 2 * eyeThickness ),
				( width * ( 0.5 - eyeOffset ) ) + eyeThickness, eyeHeight + ( 2 * eyeThickness ),
				fill = '', outline = '' ) )
				
			self.muteShapes.append( self.create_oval(
				( width * ( 0.5 + eyeOffset ) ) - eyeThickness, eyeHeight - ( 2 * eyeThickness ),
				( width * ( 0.5 + eyeOffset ) ) + eyeThickness, eyeHeight + ( 2 * eyeThickness ),
				fill = '', outline = '' ) )
				
			noseY = 0.61
			self.muteShapes.append( self.create_oval(
				( width - self.shapeThickness ) * 0.5, ( height - self.shapeThickness ) * noseY,
				( width + self.shapeThickness ) * 0.5, ( height + self.shapeThickness ) * noseY,
				fill = '', outline = '' ) )
				
			mouthCenterX = int( width / 2 )
			mouthCenterY = height * 0.7
			mouthHalfWidth = int( width * 0.125 )
			mouthHalfHeight = int( height * 0.02 )
			mouthThickness = self.shapeThickness / 2
			mouthPoints = [
				mouthCenterX, mouthCenterY - mouthThickness,
				mouthCenterX + mouthHalfWidth, mouthCenterY - mouthHalfHeight - mouthThickness,
				mouthCenterX + mouthHalfWidth + mouthThickness, mouthCenterY - mouthHalfHeight,

				mouthCenterX + mouthThickness, mouthCenterY,
				mouthCenterX + mouthHalfWidth + mouthThickness, mouthCenterY + mouthHalfHeight,
				mouthCenterX + mouthHalfWidth, mouthCenterY + mouthHalfHeight + mouthThickness,

				mouthCenterX, mouthCenterY + mouthThickness,
				mouthCenterX - mouthHalfWidth, mouthCenterY + mouthHalfHeight + mouthThickness,
				mouthCenterX - mouthHalfWidth - mouthThickness, mouthCenterY + mouthHalfHeight,

				mouthCenterX - mouthThickness, mouthCenterY,
				mouthCenterX - mouthHalfWidth - mouthThickness, mouthCenterY - mouthHalfHeight,
				mouthCenterX - mouthHalfWidth, mouthCenterY - mouthHalfHeight - mouthThickness,
			]
			# mouthPoints = [
				# mouthCenterX, mouthCenterY,
				# mouthCenterX + mouthHalfWidth, mouthCenterY - mouthHalfHeight - mouthThickness,
				# mouthCenterX + mouthHalfWidth + mouthThickness, mouthCenterY - mouthHalfHeight,

				# mouthCenterX, mouthCenterY,
				# mouthCenterX + mouthHalfWidth, mouthCenterY + mouthHalfHeight + mouthThickness,
				# mouthCenterX + mouthHalfWidth + mouthThickness, mouthCenterY + mouthHalfHeight,

				# mouthCenterX, mouthCenterY,
				# mouthCenterX - mouthHalfWidth, mouthCenterY + mouthHalfHeight + mouthThickness,
				# mouthCenterX - mouthHalfWidth - mouthThickness, mouthCenterY + mouthHalfHeight,

				# mouthCenterX, mouthCenterY,
				# mouthCenterX - mouthHalfWidth, mouthCenterY - mouthHalfHeight - mouthThickness,
				# mouthCenterX - mouthHalfWidth - mouthThickness, mouthCenterY - mouthHalfHeight,
			# ]
			self.muteShapes.append( self.create_polygon( mouthPoints, fill = '' ) )
				
		self.updateCanvas()
	
	def clearShapes( self ):
		self.delete( 'all' )
		self.micShape = None
		self.speakerShape = None
		self.muteShapes = None
		
	def onResize( self, event ):
		self.clearShapes()
		self.createShapes()

	def updateCanvas( self, event = None ):
		micTalkLevel = self.micMonitor.rollingPowerWindowEvaluator.getTalkLevelEstimate()
		speakerTalkLevel = 0
		if self.speakerMonitor is not None:
			speakerTalkLevel = self.speakerMonitor.rollingPowerWindowEvaluator.getTalkLevelEstimate()
		
		muted = micTalkLevel < 0.1 and speakerTalkLevel < 0.1
		speakerColor = ''
		if muted:
			self.configure( bg = self.muteBgColor )
			micColor = ''
			muteFgColor = self.muteFgColor
		else:
			self.configure( bg = self.bgColor )
			muteFgColor = ''
			micAlpha = self.micMonitor.rollingPowerWindowEvaluator.getTalkLevelEstimate()
			micColor = self.getAudioColor( micAlpha )
			if self.speakerMonitor is not None:
				speakerAlpha = self.speakerMonitor.rollingPowerWindowEvaluator.getTalkLevelEstimate()
				speakerColor = self.getAudioColor( speakerAlpha )
				
		self.itemconfig( self.micShape, fill = micColor )
		self.itemconfig( self.speakerShape, fill = speakerColor )

		for s in self.muteShapes:
			self.itemconfig( s, fill = muteFgColor )
			
		self.after( 300, self.updateCanvas )

	def getAudioColor( self, audioAlpha ):
		return '#{0:02x}{1:02x}{2:02x}'.format(
			self.alpha( int( self.bgColor[1:3], 16 ), int( self.fgColor[1:3], 16 ), audioAlpha ),
			self.alpha( int( self.bgColor[3:5], 16 ), int( self.fgColor[3:5], 16 ), audioAlpha ),
			self.alpha( int( self.bgColor[5:7], 16 ), int( self.fgColor[5:6], 16 ), audioAlpha ) )
	
	def alpha( self, current, overlay, alpha ):
		return min( 255, max( 0, int( alpha * overlay + ( 1 - alpha ) * current ) ) )
		
shapeMargin = 10

##### MAIN ####

import argparse

argParser = argparse.ArgumentParser( description='Brightly Display Indication of Active Voice(s)' )
argParser.add_argument('-l', dest='listDevicesAndExit', action='store_true', default=False, help='set this flag to just list the available input devices without starting the audio monitor or displaying a window' )
argParser.add_argument('-m', dest='prefMicInput', help='name of preferred input device for Microphone (partial name works)' )
argParser.add_argument('-s', dest='prefSpeakerMonitorInput', help='name of preferred input device for monitoring Speaker Output  (partial name works)' )
argParser.add_argument('-normalWindow', dest='normalWindow', action='store_true', default=False, help='In MS Windows, open as a normal window instead of Tool on top of everything')
argParser.add_argument('-g', dest='windowGeometry', default='400x800+1510+0', help='Window geometry in format "[width]x[height]+[x]+[y]", default: 400x800+1510+0')
argParser.add_argument('-bgColor', dest='bgColor', default='#111166', help='Background color of window while someone is talking (most usual web colors work including #rgb and #rrggbb)' )
argParser.add_argument('-fgColor', dest='fgColor', default='#ffff33', help='Foreground color of window while someone is talking (most usual web colors work including #rgb and #rrggbb)' )
argParser.add_argument('-mbgColor', dest='muteBgColor', default='#11ff11', help='Background color of window while nobody is talking (most usual web colors work including #rgb and #rrggbb)' )
argParser.add_argument('-mfgColor', dest='muteFgColor', default='black', help='Foreground color of window while nobody is talking (most usual web colors work including #rgb and #rrggbb)' )
args = argParser.parse_args()

micInputSearchStrings = [
	'Microsoft Sound Mapper',
	'Microphone',
	'Line In'
]

speakerMonitorInputSearchStrings = [
	'Mix',
	'Soundflower'
]

audioInterface = pyaudio.PyAudio()
micMonitor = None
speakerMonitor = None
try:
	inputDeviceInfos = []
	for i in range( audioInterface.get_device_count() ):
		info = audioInterface.get_device_info_by_index( i )
		if info[ 'maxInputChannels' ] > 0:
			inputDeviceInfos.append( info )

	if len( inputDeviceInfos ) < 1:
		print( 'No input devices!' )
		raise SystemError()
		
	print( '' )
	print( 'Available Input Devices:' )
	print( '' )
	for info in inputDeviceInfos:
		print( info[ 'name' ] )
	print( '' )

	if args.listDevicesAndExit:
		raise SystemExit()

	print( 'Default guesses will be made unless you specified inputs on the command line' )
	print( '' )
		
	micDeviceInfo = GetPreferredDeviceByName( inputDeviceInfos, args.prefMicInput, micInputSearchStrings, True )
	if args.prefMicInput is not None and not args.prefMicInput.lower() in micDeviceInfo[ 'name' ].lower():
		print( 'Your preferred Mic device ( {0} ) is not available'.format( args.prefMicInput ) )
	print( 'Mic Device Chosen: {0}'.format( micDeviceInfo[ 'name' ] ) )
	micDeviceIndex = micDeviceInfo[ 'index' ]
	
	speakerDeviceInfo = GetPreferredDeviceByName( inputDeviceInfos, args.prefSpeakerMonitorInput, speakerMonitorInputSearchStrings )
	if args.prefSpeakerMonitorInput is not None and not args.prefSpeakerMonitorInput.lower() in speakerDeviceInfo[ 'name' ].lower():
		print( 'Your preferred Speaker Output monitoring device ( {0} ) is not available'.format( args.prefSpeakerMonitorInput ) )
	print( 'Speaker Output Monitoring Device Chosen: {0}'.format( speakerDeviceInfo[ 'name' ] ) )
	speakerDeviceIndex = speakerDeviceInfo[ 'index' ]
	
	micMonitor = AudioMonitor( audioInterface, micDeviceIndex )
	micMonitor.start()

	speakerMonitor = AudioMonitor( audioInterface, speakerDeviceIndex )
	speakerMonitor.start()

	root = Tk.Tk()
	root.geometry(args.windowGeometry)
	root.title('talxlat')
	canvas = TalxlatCanvas(root, micMonitor, speakerMonitor, args.bgColor, args.fgColor, args.muteBgColor, args.muteFgColor, shapeMargin, background=args.bgColor)
	canvas.pack( fill = Tk.BOTH, expand = Tk.YES )
	if not args.normalWindow:
		root.wm_attributes( '-topmost', 1 )
		root.wm_attributes( '-toolwindow', 1 )

	root.mainloop()
finally:
	if micMonitor is not None:
		micMonitor.stop()
	if speakerMonitor is not None:
		speakerMonitor.stop()
	audioInterface.terminate()
	
# Console monitor
# audioInterface = pyaudio.PyAudio()
# try:
	# for i in range( audioInterface.get_device_count() ):
		# print( audioInterface.get_device_info_by_index(i) )

	# micDeviceIndex = 1

	# micMonitor = AudioMonitor( audioInterface, micDeviceIndex )
	# micMonitor.start()

	# while True:
		# time.sleep( 1 )
		# mute = ''
		# if micMonitor.muteDetected:
			# mute = ' (Muted)'
		# rollingPowerWindowEvaluator = micMonitor.rollingPowerWindowEvaluator
		# print( '{0}{1}'.format( '\t'.join( map( lambda x: str( x ), [ 
			# # rollingPowerWindowEvaluator.totalOutlyingPower,
			# # rollingPowerWindowEvaluator.getOutlyingPowerPercent(),
			# rollingPowerWindowEvaluator.getTalkLevelEstimate(),
			# rollingPowerWindowEvaluator.getTotalPowerPercent(),
			# rollingPowerWindowEvaluator.getPowerCoefficientOfVariation(),
			# rollingPowerWindowEvaluator.outlyingPowerThreshhold,
			# rollingPowerWindowEvaluator.powerStdDevEstimateTimesBufferSize / rollingPowerWindowEvaluator.powerWindowBuffer.size,
			# rollingPowerWindowEvaluator.totalPower ] ) ),
			# mute ) )
		
# finally:
	# if micMonitor is not None:
		# micMonitor.stop()
	# audioInterface.terminate()


# roll test
# rollingPowerWindowEvaluator = RollingPowerWindowEvaluator( 50, 5, Int16Settings() )
# n = 0
# while True:
	# newData = numpy.zeros( 16, dtype = numpy.int16 )
	# newData[ 5 ] = 11
	# newData[ 9 ] = 11
	# newData[ 15 ] = 11
# #	newData = numpy.arange( n * 16, ( n+1 ) * 16, dtype = numpy.int16 )
	# rollingPowerWindowEvaluator.addData( newData )
	# n += 1
	# print( newData )
	# print( rollingPowerWindowEvaluator.powerWindowBuffer )
	# print( '\t'.join( map( lambda x: str( x ), [ rollingPowerWindowEvaluator.totalOutlyingPower, rollingPowerWindowEvaluator.getOutlyingPowerPercent(), rollingPowerWindowEvaluator.outlyingPowerThreshhold, rollingPowerWindowEvaluator.totalSquaresOfPower, rollingPowerWindowEvaluator.powerStdDevEstimateTimesBufferSize, rollingPowerWindowEvaluator.totalPower ] ) ) )
	# time.sleep( 2 )
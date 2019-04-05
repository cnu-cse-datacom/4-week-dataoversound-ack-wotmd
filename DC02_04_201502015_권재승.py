from __future__ import print_function

import sys
import wave
import pyaudio
import numpy as np

from io import StringIO

import alsaaudio
import colorama
import numpy as np

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format
from time import sleep

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4

STUDENT_ID = "201502015"

def stereo_to_mono(input_file, output_file):
	inp = wave.open(input_file, 'r')
	params = list(inp.getparams())
	params[0] = 1 # nchannels
	params[3] = 0 # nframes

	out = wave.open(output_file, 'w')
	out.setparams(tuple(params))

	frame_rate = inp.getframerate()
	frames = inp.readframes(inp.getnframes())
	data = np.fromstring(frames, dtype=np.int16)
	left = data[0::2]
	out.writeframes(left.tostring())

	inp.close()
	out.close()

def yield_chunks(input_file, interval):
	wav = wave.open(input_file)
	frame_rate = wav.getframerate()

	chunk_size = int(round(frame_rate * interval))
	total_size = wav.getnframes()

	while True:
		chunk = wav.readframes(chunk_size)
		if len(chunk) == 0:
			return

		yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
	w = np.fft.fft(chunk)
	freqs = np.fft.fftfreq(len(chunk))	# double -> int
	peak_coeff = np.argmax(np.abs(w))	
	peak_freq = freqs[peak_coeff]		# dominant freq
	return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
	return abs(freq1 - freq2) < 20

# 4bit 4bit -> 1byte ascii
def decode_bitchunks(chunk_bits, chunks):
	out_bytes = []

	next_read_chunk = 0
	next_read_bit = 0

	byte = 0
	bits_left = 8
	while next_read_chunk < len(chunks):
		can_fill = chunk_bits - next_read_bit
		to_fill = min(bits_left, can_fill)
		offset = chunk_bits - next_read_bit - to_fill
		byte <<= to_fill
		shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
		byte |= shifted >> offset;
		bits_left -= to_fill
		next_read_bit += to_fill
		if bits_left <= 0:

			out_bytes.append(byte)
			byte = 0
			bits_left = 8

		if next_read_bit >= chunk_bits:
			next_read_chunk += 1
			next_read_bit -= chunk_bits

	return out_bytes
	
def encode_bitchunks(chunk_bits, plain_bytes):
	chunks = []
	for one_byte in plain_bytes:
		temp_chunks = []
		for i in range(0, 8//chunk_bits):
			shift = (chunk_bits*i)
			t_chunks = (one_byte>>shift) & 0xF
			temp_chunks.append(t_chunks)
		chunks.append(temp_chunks[::-1])
	return chunks

def decode_file(input_file, speed):
	wav = wave.open(input_file)
	if wav.getnchannels() == 2:
		mono = StringIO()
		stereo_to_mono(input_file, mono)

		mono.seek(0)
		input_file = mono
	wav.close()

	offset = 0
	for frame_rate, chunk in yield_chunks(input_file, speed / 2):
		dom = dominant(frame_rate, chunk)
		print("{} => {}".format(offset, dom))
		offset += 1

def extract_packet(freqs):
	freqs = freqs[::2]	# 1 1 2 2 3 3 => 1 2 3
	bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
	bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
	return bytearray(decode_bitchunks(BITS, bit_chunks))
	
def make_packet(bitchunks):
	freqs = [(bit * STEP_HZ + START_HZ) for bit in bitchunks]
	freqs.insert(0, HANDSHAKE_START_HZ)
	freqs.insert(0, HANDSHAKE_START_HZ)
	freqs.append(HANDSHAKE_END_HZ)
	freqs.append(HANDSHAKE_END_HZ)
	return freqs

def display(s):
	cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')


def speaking_linux(byte_stream, frame_rate=44100, interval=0.1):
	p = pyaudio.PyAudio()
	duration = 0.5
	stream = p.open(format=pyaudio.paFloat32, channels=1, rate=frame_rate, output = True)
	#print("speak")
	#print(byte_stream)
	raw_bitchunks = encode_bitchunks(BITS, byte_stream)
	bitchunks = []
	for raw in raw_bitchunks:
		bitchunks.append(raw[0])
		bitchunks.append(raw[1])
	frequency = make_packet(bitchunks)

	## dummy
	samples = (np.sin(2*np.pi*np.arange(frame_rate*duration)*4096/frame_rate)).astype(np.float32)
	stream.write(samples)
	
	for f in frequency:
		samples = (np.sin(2*np.pi*np.arange(frame_rate*duration)*f/frame_rate)).astype(np.float32)
		stream.write(samples)
		print("freq : " + str(f))
	stream.stop_stream()
	stream.close()
	
	p.terminate()


def listen_linux(frame_rate=44100, interval=0.1):

	mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
	mic.setchannels(1)	# chanel 1 use
	mic.setrate(44100)	# frame_rate   44100/s
	mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

	num_frames = int(round((interval / 2) * frame_rate))
	mic.setperiodsize(num_frames)	#
	print("start...")

	in_packet = False
	packet = []

	while True:
		l, data = mic.read()
		if not l:
			continue

		chunk = np.fromstring(data, dtype=np.int16)
		dom = dominant(frame_rate, chunk)

		if in_packet and match(dom, HANDSHAKE_END_HZ):
			print("end")
			byte_stream = extract_packet(packet)
			try:	# 4 bytes
				byte_stream = RSCodec(FEC_BYTES).decode(byte_stream) # reed solo error check
				byte_stream = byte_stream.decode("utf-8")
				if STUDENT_ID in byte_stream:
					byte_stream = byte_stream.replace(STUDENT_ID, "").strip()
					display(byte_stream)
					byte_stream = RSCodec(FEC_BYTES).encode(byte_stream)
					sleep(1)
					speaking_linux(byte_stream)
					sleep(1)
			except ReedSolomonError as e:
				pass
				#print("{}: {}".format(e, byte_stream))
			packet = []
			in_packet = False
		elif in_packet:
			packet.append(dom)
		elif match(dom, HANDSHAKE_START_HZ):
			in_packet = True
			print("start HandShake")

if __name__ == '__main__':
	colorama.init(strip=not sys.stdout.isatty())

	#decode_file(sys.argv[1], float(sys.argv[2]))
	listen_linux()

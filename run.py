from pathlib import Path
from argparse import ArgumentParser

from faster_whisper import WhisperModel

def parse_args():
	parser = ArgumentParser('音声ファイルから文字起こしを行います.')
	parser.add_argument('filename', type=Path, help='入力音声ファイル')
	parser.add_argument('--out_path', type=Path, default=None, help='出力パス')
	parser.add_argument('--model_name', type=str, default='large-v3', help='使用するwisperのmodel名')
	parser.add_argument('--use_vad_filter', action='store_true', help='音声フィルタを使うか')
	return parser.parse_args()

def transcribe(file_path, model='large-v3' ,use_vad_filter=False):
	result = []

	model = WhisperModel(model, device="cpu", compute_type="int8")
	# Run on GPU with FP16
	#model = WhisperModel(model_size, device="cuda", compute_type="float16")
	# or run on GPU with INT8
	# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
	# or run on CPU with INT8
	# model = WhisperModel(model_size, device="cpu", compute_type="int8")

	segments, info = model.transcribe(str(file_path.resolve()), beam_size=3, language="ja")

	print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

	for segment in segments:
		output = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
		print(output)
		result.append(output)
	return result

if __name__ == '__main__':
	args = parse_args()
	output = transcribe(args.filename, args.model_name, args.use_vad_filter)
	if args.out_path is not None:
		with args.out_path.open('w') as f:
			f.write('\n'.join(output))

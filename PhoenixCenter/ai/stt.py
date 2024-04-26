import time
import threading

import sounddevice as sd
import sherpa_ncnn


class Stt:
    @staticmethod
    def nullptr(*args, **kwargs):
        pass

    def __init__(self, model="sherpa-ncnn-en-20M") -> None:
        self.model = model
        self.resolve = Stt.nullptr
        self.haveResolve = False
        self._exit = False

    def then(self, resolve) -> None:
        self.resolve = resolve
        self.haveResolve = True

    def create_recognizer(self, model=None) -> sherpa_ncnn.Recognizer:
        # Please replace the model files if needed.
        # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
        # for download links.
        # MODEL_DIR = "ai/model/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23"
        MODEL_DIR = f"ai/model/{model or self.model or 'sherpa-ncnn-en-20M'}"
        recognizer = sherpa_ncnn.Recognizer(
            tokens=f"{MODEL_DIR}/tokens.txt",
            encoder_param=f"{MODEL_DIR}/encoder_jit_trace-pnnx.ncnn.param",
            encoder_bin=f"{MODEL_DIR}/encoder_jit_trace-pnnx.ncnn.bin",
            decoder_param=f"{MODEL_DIR}/decoder_jit_trace-pnnx.ncnn.param",
            decoder_bin=f"{MODEL_DIR}/decoder_jit_trace-pnnx.ncnn.bin",
            joiner_param=f"{MODEL_DIR}/joiner_jit_trace-pnnx.ncnn.param",
            joiner_bin=f"{MODEL_DIR}/joiner_jit_trace-pnnx.ncnn.bin",
            num_threads=4,
        )
        return recognizer

    def start(self):
        def reco():
            print("Recognizer Started!")
            recognizer = self.create_recognizer()
            sample_rate = recognizer.sample_rate
            samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

            last_result = ""
            with sd.InputStream(
                channels=1, dtype="float32", samplerate=sample_rate
            ) as s:
                start_time = time.time()
                while True:
                    if self._exit:
                        break

                    while not self.haveResolve:
                        time.sleep(0.01)

                    samples, _ = s.read(samples_per_read)  # a blocking read
                    samples = samples.reshape(-1)
                    recognizer.accept_waveform(sample_rate, samples)
                    result = recognizer.text

                    if last_result != result:
                        last_result = result
                        start_time = time.time()

                    if (
                        result
                        and last_result == result
                        and time.time() - start_time > 0.7
                    ):
                        # 0.7s of quiet
                        self.resolve(result)

                        recognizer.reset()
                        start_time = time.time()
                        self.resolve = Stt.nullptr
                        self.haveResolve = False

        self.thread = threading.Thread(target=reco)
        self.thread.start()

    def exit(self):
        self._exit = True


if __name__ == "__main__":
    stt = Stt()
    stt.resolver = print
    stt.start()
    time.sleep(1)
    print("Start ...")
    stt.then(lambda x: (print(x), stt.exit()))

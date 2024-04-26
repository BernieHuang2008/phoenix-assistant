from gtts import gTTS
import os


def say(text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")


if __name__=='__main__':
    say("Hello Bernie. This is Phoenix System at your service.")


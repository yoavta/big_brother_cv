from subprocess import call


def speak(text):
    cmd_beg = 'pico2wave -w testpico.wav "'
    cmd_end = '" && paplay testpico.wav'  # To play back the stored .wav file and to dump the std errors to /dev/null
    call([cmd_beg + text + cmd_end], shell=True)
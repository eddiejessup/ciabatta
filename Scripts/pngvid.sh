mencoder mf://*.png -mf w=800:h=600:fps=50:type=png -ovc x264 -x264encopts crf=25 -oac copy -o output.avi

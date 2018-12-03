GOAL = ./ocr
CFLAGS = -Wall -g
# LIBS = -ltiff `pkg-config --libs gtk+-2.0 fribidi`
LIBS = -lefence -lm # for debugging
LIBS = -lm
# LIBS += -ltiff `pkg-config --libs gtk+-2.0` fribidi/libfribidi.`uname -m`.so
BIDI = /usr/lib/x86_64-linux-gnu/libfribidi.so.0
PANGOX = /usr/lib/x86_64-linux-gnu/libpangox-1.0.so.0
LIBS += -ltiff `pkg-config --libs gtk+-2.0` $(BIDI) $(PANGOX)
# LIBS += /tmp/dmalloc-5.5.2/libdmalloc.a
OBJS = readPicture.o main.o segment.o display.o gtkDisplay.o categorize.o \
	training.o template.o dl.o
JSONS = *.json *.h5
INCLUDES = `pkg-config --cflags gtk+-2.0`
PYTHON = -lpython3.6m -L /usr/lib/python3.6/config/ -I /usr/include/python3.6/
OLDINCLUDES = -I /usr/include/gtk-2.0 -I /usr/include/glib-2.0 \
	-I /usr/lib/x86_64-linux-gnu/glib-2.0/include \
	-I /usr/lib/x86_64-linux-gnu/gtk-2.0/include \
	-I /usr/include/cairo \
	-I /usr/include/pango-1.0 \
	-I /usr/include/gdk-pixbuf-2.0 \
	-I /usr/include/atk-1.0
.SUFFIXES: .pdf .ps .tiff .tif .uyid
OCR = ./ocr

default: $(GOAL)

.c.o:
	gcc $(CFLAGS) $(INCLUDES) -c $*.c

$(GOAL): $(OBJS) tags
	gcc $(OBJS) $(LIBS) $(PYTHON) -o $(GOAL)

$(OBJS): ocr.h

tags: *.c *.h
	ctags *.c *.h

clean:
	rm -f $(GOAL) $(OBJS) *.pyc $(JSONS)
	rm -f -r __pycache__/

IMPORTANT = ocr/*.c ocr/*.h ocr/demo ocr/../fontData/yiddishData \
	ocr/README ocr/Makefile ocr/COPYRIGHT ocr/fixutf8.pl

DOWNLOAD = *.c *.h demo fontData README Makefile COPYRIGHT fridibi fixutf8.pl

tar:
	cd ..; tar hczf ocr.tar.gz $(IMPORTANT)

upload:
	cd ..; rsync -a $(IMPORTANT) bud:projects/ocr

download:
	rsync -a `echo "bud:projects/ocr/{$(DOWNLOAD)}" | tr ' ' ,` .

# a sample to show how to do it
interactive: ocr
	ocr -f ../fontData/yiddishData demo/nybc200960_0100

# a sample to show how to do it
batch: ocr
	./ocr -f ../fontData/yiddishData -t demo/nybc200960_0100 | perl fixutf8.pl

burstPDF:
	pdftk $(PDF) burst

# a sample to show how to do it.
TIFFS = $(subst .pdf,.tiff,$(wildcard tmp/raismann/*.pdf))
tiffs: $(TIFFS)

# .pdf.tiff:
#	 convert -density 200 -threshold 75% -compress LZW $*.pdf $*.tiff

.pdf.ps:
	ps2pdf $*.pdf $*.ps

.ps.tiff:
	echo '' | gs -sDEVICE=tiffgray -sOutputFile=$*.tiff -r400x400 $*.ps > /dev/null

# the following works very nicely
.pdf.tiff:
	echo '' | \
		gs -sDEVICE=tiffgray -sOutputFile=$*.tiff -sCompression=lzw \
			-r400x400 $*.pdf \
		> /dev/null
	tiffsplit $*.tiff page_
	@echo Please move page_* to the directory of $*

# rotate clockwise
rotate:
	for i in $(DIR)/*.tif* ; do \
		echo $$i; \
		tiffcrop -R 90 $$i tmp.tiff; \
		mv tmp.tiff $$i; \
	done

BASHEVIS = -H 600 -W 600 -d 30 -w 3 -h 5 -x -i -p 1.4 -s 0.4 -A -m 1.1
bashevis-i:
	$(OCR) -f ../fontData/bashevis.data $(BASHEVIS) \
	-b bashevis ../tmp/bashevis-sotn-in-goray/page*.tif

bashevis-b:
	$(OCR) -f ../fontData/bashevis.data $(BASHEVIS) -t \
	-b bashevis	../tmp/bashevis-sotn-in-goray/page*.tif

bashevis-t:
	python3 dl_train.py bashevis.data

# Lorem Ipsum fontData example
TEST = -H 600 -W 600 -d 15 -w 3 -h 4 -p 1.2 -m 1.0 -x -i
test-i:
	$(GOAL) -f ../fontData/english.data $(TEST) \
	-b english ../samples/english.tif

test-b:
	$(GOAL) -f ../fontData/english.data $(TEST) -t \
	-b english ../samples/english.tif

# Train the Deep Network
test-t:
	python3 dl_train.py english.data

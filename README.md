# Deep Learning OCR #
This program is an extension of Dr. Raphael Finkel’s original Optical Character Recognition program.
Our group implemented a deep learning method and a support vector machine method into the existing code.
The Optical Character Recognition program is designed to train itself in order to be able to predict a scanned tiff image file and make a mostly accurate textual prediction of the file. 

## Authors ##
- Dr. Raphael Finkel
- John Geddes
- Greg McIntosh
- Taylor Goldhahn

### Prerequisites ###
These methods require multiple libraries in order to run the Python functions within the program.
There is a setup script set up to help download all libraries needed to run with this program.
This can be done by running “chmod u+x setup.sh” and then “./setup.sh” while in the “combined” folder of this project.
Make sure that once it installs Python 3.6, make it the current system Python version. Also, if your CPU is too old, then you must modify the setup script and read the comments at the bottom.
The most recent versions of TensorFlow only work on newer CPUs, and you must uninstall and reinstall TensorFlow, but in version 1.5 as described in the script’s comments.

### Running the Program ###
A makefile has been written to quickly run the program on certain files.
In a terminal in the "combined" directory, run “make default”. After that, you have the choice to run whatever version of the makefile is needed.
In the Makefile, the examples are in the form of “make filename-mode-method” where filename can be ‘english’ for a “Lorem Ipsum” example, or ‘bashevis’ for the “Bashevis” example.
The ‘mode’ can be set to ‘t’ for the training mode, ‘b’ for the command line prediction mode, or ‘i’ for the interactive mode.
Currently only the KD trees method works with the interactive mode.
The ‘method’ can be set to ‘KD’ for the original kd tree method, ‘DL’ for the newly implemented deep learning method, or ‘SVM’ for the newly implemented support vector machine method.

#### Example ####
If you want to run the DL method after downloading this directory, do the following:
``` bash
make default
make bashevis-t-DL
make bashevis-b-DL
```
The first line compiles the project code.
The second line trains the Deep Learning network on the Bashevis book and saves the network to a few files.
The third line runs the Deep Learning prediction method on the Bashevis book and outputs the result to the terminal.


If you would like to run a new file, you may run the OCR program with the following parameter form:

Usage: ./ocr -f fontData -KDV [-t] [-h n] [-w n] [-s n] [-W n] [-H n] [image ...]
  - image, image.tif, or image.tiff is the image file
  - image.training is its training file.
  - fontData associates glyph statistics with UTF8 strings.
  - Choose the training and predicting method:
    - -K for Dr. Finkel's K-Dimension Tree method
    - -D for our Deep Learning prediction method
    - -V for our Support-Vector Machine method
  - -c n for n-column input.
  - -t causes text output; it does not have an interactive component
  - -h n only considers glyphs at least n pixels tall (default %d)
  - -w n only considers glyphs at least n pixels wide (default %d)
  - -H n only considers glyphs at most n pixels high (default %d)
  - -W n only considers glyphs at most n pixels wide (default %d)
  - -m n max distance to consider a match (default %3.2f)\n"
  - -g n max distance to consider a good match (ordinary rectangle) (default %3.2f)
  - -p n an unrecognized glyph this much wider than normal might split (default %3.2f)
  - -s n a space must be at least this fraction of average glyph width (default %3.2f)
  - -C n cutoff at this percent of full black (default %3.2f)
  - -S do not try to shear to correct for rotation
  - -L n use n as the italic correction; Δx = Δy/slant. Bigger is more vertical. (default %3.2f)
  - -x use flooding to determine glyph boundaries.
  - -X do not try to combine or split.
  - -A always combine horizontal overlaps, even if result is worse
  - -i ignore glyph's vertical placement in matching glyphs
  - -d n minimum glyph area

Removing the -t parameter will start the interactive mode of this program.
To run training run the command, “python3 dl_train.py [datafile]” or “python3 svm_train.py [datafile]”, where datafile is a fontdata file that is to be trained.

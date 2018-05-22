# cytonNss
Cyton Online Neural Sentence Segmentation for Simultaneous Interpretation

This software implments an Online Sentence Segmentation for Simultaneous Interpretation using Multi-Shifted Recurrent Neural Network
!! Note: the newlines of the txt files are in Linux format!!

Xiaolin Wang (xiaolin.wang@nict.go.jp, arthur.xlw@gmail.com)


The software is coded in C++ with minimal dependencies:

	CUDA >= 8.0
	CUDNN >= 7.0

Please run the script of run.sh for compilation, training, tuning and test.
You can also try each step manually through the following commands.

If you are using our toolkit, please kindly cite our paper (to appear).

=====================================================================================

# 1) Compilation:

make -j8;

# 2)  Training:

cd data
../bin/cytonNss --mode train --train train.txt --vocabFile train.vocab.txt --dev dev.txt --saveModel model --hiddenSize 128 --numLayers 2  

Note: You may increase the size of neural network for high performance.


# 3) Tuning:

 ../bin/cytonNss --mode tune --dev dev.txt --loadModel model/model --output tune  --tuneSteps 100

Note: You may increase the tuneSteps for high performance.
 

# 4) Test

../bin/cytonNss --mode apply --input stdin --output stdout --loadModel model/model --thresholds 0.9:0.8:0.7:0.6:0.5:0.4 testInput.txt > testOutput.txt

Note: Please set the thresholds according to the results of tuning.

# Format of Input: 

The input file contains sessions separated by empty line.
For example:
	are you		% a few input words from the 1-st session
	okay            % more input words from the 1-st session
	i 'm            % more input words from the 1-st session
	okay            % more input words from the 1-st session

		        % empty line, end of the 1-st session
		  
i 'm              <-- input words from 2-th session

tired             <-- more input words from the 2-nd session

how do            <-- more input words from the 2-nd session

you do            <-- more input words from the 2-nd session

what is           <-- more input words from the 2-nd session

your name         <-- more input words from the 2-nd session

yes               <-- more input words from the 2-nd session
		
		  <--- empty line, end of the 2-nd session


# Format of Ouput: one sentence per line, and sessions are separated by empty line


are you okay      % sentence 0

i 'm okay         % sentence 1

		 <--- empty line, end of the 1-st session
i 'm tired        <--- sentence 0
how do you do     <--- sentence 1
what is your name <--- sentence 2
yes               <--- sentence 3
                  <--- empty line, end of the 2-nd session
=====================================================================================
# USAGE
../bin/cytonNss  --help
--help	 ()
--mode	Running mode: train/tune/apply (train)
--loadModel	Load model from a model directory ()
--saveModel	Save model to a model directory ()
--vocabSize	Size of the vocabulary. 0 means unlimited. (0)
--hiddenSize	Size of hidden states (512)
--numLayers	Number of layers (3)
--optimizer	Optimizer (SGD)
--learningRate	Learning rate (1.0)
--learnRateDecay	Decay rate of learning rate (0.5)
--dropout	Dropout rate (0.5)
--maxSentLen	Maximun length of sentence (40)
--numFutureWords	Maximun number of future words (6)
--vocabFile	Vocabulary File (trainFile.vocab)
--train	Training file. One sentence per line. (trainFile)
--dev	Ddevelopment file. One sentence per line. (dev)
--batchSize	Batch size (64)
--input	Input File of the apply mode. stdin means standard input ()
--output	Output File of the apply mode or the tune mode. stdin means standard output ()
--thresholds	Thresholds (0.9:0.8:0.7:0.6:0.5:0.4)
--scoreTolerance	Tolerance of scores in the tune mode (0.04)
--tuneSteps	Maximum number of tune steps. O means infinite. (0)
--factorLatency	Factory of latency in evaluation: score=f1-factorLatency*latency. (0.01)
--numSents	Number of sentences as one input in the tune mode (10)




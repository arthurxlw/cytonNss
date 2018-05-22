#1) compiling the toolkit;
make -j8;

#2) training
# You may increase the size of neural network for high performance
cd data
../bin/cytonNss --mode train --train train.txt --vocabFile train.vocab.txt --dev dev.txt --saveModel model --hiddenSize 128 --numLayers 2  

#3) tuning
# You may increase the tuneSteps for high performance
../bin/cytonNss --mode tune --dev dev.txt --loadModel model/model --output tune  --tuneSteps 100

#3) test
# Please set the thresholds according to the results of tuning
../bin/cytonNss --mode apply --input stdin --output stdout --loadModel model/model --thresholds 0.9:0.8:0.7:0.6:0.5:0.4 <testInput.txt >testOutput.txt


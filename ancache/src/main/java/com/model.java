import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.nlp.Squad;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.CheckpointsTrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.training.listener.TrainingListener;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SquadLSTMExample {

    public static void main(String[] args) throws IOException, TranslateException {
        // Set the path to save the model
        Path modelDir = Paths.get("build/squad_lstm");

        // Load the SQuAD dataset
        Squad squad = Squad.builder()
                .optUsage(Squad.Usage.TRAIN)
                .setSampling(32, true)
                .build();

        // Create the LSTM model
        int inputSize = 768;  // Size of input embeddings
        int hiddenSize = 256; // Size of LSTM hidden layer
        int numLayers = 1;     // Number of LSTM layers
        int numClasses = 2;    // Number of output classes (start and end positions)
        Block lstmBlock = new LSTM.Builder()
                .setNumLayers(numLayers)
                .setStateSize(hiddenSize)
                .optReturnState(true)
                .build();
        Block outputBlock = new SequentialBlock()
                .add(lstmBlock)
                .add(Linear.builder().setUnits(numClasses).build());
        
        Model model = Model.newInstance("squad-lstm");
        model.setBlock(outputBlock);

        // Create a translator to convert input data to NDArray
        Translator<TextData, NDList> translator = new Translator<TextData, NDList>() {
            @Override
            public NDList processInput(TranslatorContext ctx, TextData input) {
                NDArray inputArray = input.toNDArray(ctx.getNDManager(), DataType.FLOAT32);
                return new NDList(inputArray);
            }

            @Override
            public NDList processOutput(TranslatorContext ctx, NDList list) {
                return list;
            }
        };

        // Setup training configuration
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optOptimizer(Optimizer.adam().optLearningRateTracker(LearningRateTracker.fixedLearningRate(0.001f)))
                .optDevices(Device.getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging("squad-lstm"))
                .addTrainingListeners(new CheckpointsTrainingListener(modelDir));

        // Create the Trainer
        try (Trainer trainer = model.newTrainer(config)) {
            trainer.setBatchifier(Batchifier.STACK);
            trainer.initialize(new Shape(1, inputSize));

            // Train the model
            for (int epoch = 1; epoch <= 10; epoch++) {
                trainEpoch(trainer, squad, translator);
            }
        }
    }

    private static void trainEpoch(Trainer trainer, Dataset dataset, Translator<TextData, NDList> translator)
            throws IOException, TranslateException {
        for (Batch batch : trainer.iterateDataset(dataset)) {
            NDList data = translator.processBatch(trainer.getManager(), batch.getData());
            NDList labels = translator.processBatch(trainer.getManager(), batch.getLabels());
            trainer.trainBatch(new Batch(data, labels));
            batch.close();
            trainer.step();
        }
    }
}
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.LSTM;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.Squad;
import ai.djl.modality.nlp.qa.SquadTranslator;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.listener.TrainingListener.Defaults;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class SquadInferenceExample {

    public static void main(String[] args) throws IOException, TranslateException {
        // Set the path to the saved model directory
        Path modelDir = Paths.get("build/squad_lstm");

        // Load the saved model
        Model model = Model.newInstance("squad-lstm");
        model.load(modelDir);

        // Prepare the input data (question and context)
        String question = "Who is the author of the book 'To Kill a Mockingbird'?";
        String context = "To Kill a Mockingbird is a novel by Harper Lee published in 1960. "
                + "It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.";

        // Perform inference
        try (Predictor<TextData, List<String>> predictor = model.newPredictor(new SquadTranslator())) {
            List<String> answers = predictor.predict(new TextData(question, context));
            System.out.println("Predicted Answer(s):");
            for (String answer : answers) {
                System.out.println(answer);
            }
        }
    }
}
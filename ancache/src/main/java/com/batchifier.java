public class QAPaddingBatchifier implements Batchifier {

    private final int paddingValue;

    public QAPaddingBatchifier(int paddingValue) {
        this.paddingValue = paddingValue;
    }

    @Override
    public NDList batchify(NDList[] inputs) throws TranslateException {
        NDList batch = new NDList();

        // Check if the input length is even (assuming separate context and question)
        if (inputs.length % 2 != 0) {
            throw new TranslateException("Uneven number of inputs for QAPaddingBatchifier");
        }

        int numExamples = inputs.length / 2;

        // Separate context and question tensors
        NDList[] contextList = new NDList[numExamples];
        NDList[] questionList = new NDList[numExamples];
        for (int i = 0; i < numExamples; i++) {
            contextList[i] = inputs[2 * i];
            questionList[i] = inputs[2 * i + 1];
        }

        // Get the maximum sequence length (consider context and question separately)
        int maxContextLength = 0;
        try (NDManager manager = NDManager.newBaseManager()) { // Create NDManager instance within the method
            for (NDList context : contextList) {
                NDArray contextArray = context.get(0); // Assuming single tensor in context list
                maxContextLength = Math.max(maxContextLength, contextArray.shape().get(1));
            }
        }
        int maxQuestionLength = 0;
        try (NDManager manager = NDManager.newBaseManager()) { // Create another NDManager instance for question
            for (NDList question : questionList) {
                NDArray questionArray = question.get(0); // Assuming single tensor in question list
                maxQuestionLength = Math.max(maxQuestionLength, questionArray.shape().get(1));
            }
        }

        // Pad each context and question tensor independently
        NDList[] paddedContextList = new NDList[numExamples];
        NDList[] paddedQuestionList = new NDList[numExamples];
        try (NDManager manager = NDManager.newBaseManager()) { // Create another NDManager instance for padding
            for (int i = 0; i < numExamples; i++) {
                paddedContextList[i] = pad(manager, contextList[i].get(0), maxContextLength);
                paddedQuestionList[i] = pad(manager, questionList[i].get(0), maxQuestionLength);
            }
        }

        // Combine padded context and question into a single output structure
        for (int i = 0; i < numExamples; i++) {
            NDList combined = new NDList();
            combined.add(paddedContextList[i]);
            combined.add(paddedQuestionList[i]);
            batch.add(combined.concat(new int[]{1})); // Concatenate along sequence dimension
        }

        return batch;
    }

    private NDArray pad(NDManager manager, NDArray array, int maxLength) {
        int currentLength = array.shape().get(1);
        if (currentLength >= maxLength) {
            return array;
        }
        NDArray padding = manager.ones(new Shape(array.shape().get(0), maxLength - currentLength))
                .mul(paddingValue);
        return NDArray.concat(new NDList(array, padding), 1);
    }
}

import java.io.InputStream;

import binaryIV.BinaryIVTask;

public class testing {

    public static void main(String[] args) {
        String filePath = "D:/TUM Workspace/Thesis Workspace/causal-bounds/simulation_engine/algorithms/zaffalon/binaryIV/input_data.csv"; // Corrected to use forward slashes for cross-platform compatibility

        try (InputStream is = new java.io.FileInputStream(filePath)) {
            String query = "ATE";
            BinaryIVTask task = new BinaryIVTask(is, query);
            String result = task.call();
            System.out.println("Result: " + result);
        } catch (Exception e) {
            System.err.println("Error occurred while running the simulation.");
            e.printStackTrace();
        }
    }

}

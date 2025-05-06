package binaryIV;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class ParallelRunner {
    public static void main(String[] args) throws Exception {
        String inputCSV = "zaffalon_bounds/binaryIV/data_for_zaffalon.csv";
        String outputCSV = "zaffalon_bounds/binaryIV/results_parallel.csv";

        // Limit how many simulations to run (for testing)
        int N_INSTANCES = 10; // ← Change this to test with fewer simulations
        int START_B = -5000;
        int STEP = 5;

        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Using " + numThreads + " threads to run " + N_INSTANCES + " simulations.");

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<String>> futures = new ArrayList<>();

        for (int i = 0; i < N_INSTANCES; i++) {
            int b = START_B + i * STEP;
            futures.add(executor.submit(new BinaryIVAteSimulationTask(b, inputCSV)));
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        // Write all results to single output file
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputCSV))) {
            writer.println("b_X_Y_1000,zaffalon_bound_lower,zaffalon_bound_upper");
            for (Future<String> future : futures) {
                writer.println(future.get());
            }
        }

        System.out.println("All simulations completed.");
    }
}

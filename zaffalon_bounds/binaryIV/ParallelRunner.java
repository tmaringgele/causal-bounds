package binaryIV;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import binaryIV.BinaryIVAteSimulationTask;

public class ParallelRunner {
    public static void main(String[] args) throws Exception {
        String inputCSV = "zaffalon_bounds/binaryIV/data_for_zaffalon.csv";
        String outputCSV = "zaffalon_bounds/binaryIV/results_parallel.csv";

        // Limit how many simulations to run (for testing)
        int N_INSTANCES = 2000; // Change this to test with fewer simulations
        int START_B = -5000;
        int STEP = 5;

        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("Using " + numThreads + " threads to run " + N_INSTANCES + " simulations.");

        long startTime = System.nanoTime(); //  Start timer

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<String> completionService = new ExecutorCompletionService<>(executor);

        for (int i = 0; i < N_INSTANCES; i++) {
            int b = START_B + i * STEP;
            completionService.submit(new BinaryIVAteSimulationTask(b, inputCSV));
        }

        // Write all results as they complete
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputCSV))) {
            writer.println("b_X_Y_1000,zaffalon_bound_lower,zaffalon_bound_upper");

            for (int i = 0; i < N_INSTANCES; i++) {
                Future<String> completedFuture = completionService.take(); // blocks until next is done
                String result = completedFuture.get();
                writer.println(result);

                // Fortschritt anzeigen
                System.out.printf("Progress: %d / %d completed (%.1f%%)%n",
                        i + 1, N_INSTANCES, 100.0 * (i + 1) / N_INSTANCES);
            }
        }

        executor.shutdown();

        long endTime = System.nanoTime(); // End timer
        double durationMillis = (endTime - startTime) / 1_000_000.0;
        double durationSeconds = durationMillis / 1000.0;
        double durationMinutes = durationSeconds / 60.0;

        System.out.printf("Runtime: %.2f ms (%.2f s | %.2f min)%n", durationMillis, durationSeconds, durationMinutes);
        System.out.println("All simulations completed.");
    }
}

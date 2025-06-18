package binaryIV;
import gnu.trove.map.TIntIntMap;

import java.util.concurrent.Callable;

/**
 * Represents a single simulation task for computing ATE bounds
 * using binary instrumental variables.
 */
public class BinaryIVAteSimulationTask implements Callable<String> {
    private final int b_X_Y_1000;
    private final String inputCSV;

    public BinaryIVAteSimulationTask(int b_X_Y_1000, String inputCSV) {
        this.b_X_Y_1000 = b_X_Y_1000;
        this.inputCSV = inputCSV;
    }

    @Override
    public String call() {
        int Z = 0, X = 1, Y = 2, U = 3;

        try {
            TIntIntMap[] data = BinaryIVBounds.getDataFromCSV(inputCSV, Z, X, Y, b_X_Y_1000);
            double[] bounds = BinaryIVBounds.getBoundsForBinaryIV(data, 100, 30, "ate", Z, X, Y, U);

            return String.format("%d,%.4f,%.4f", b_X_Y_1000, bounds[0], bounds[1]);
        } catch (Exception e) {
            System.err.println("Simulation failed for b_X_Y_1000 = " + b_X_Y_1000);
            e.printStackTrace();
            return String.format("%d,ERROR,ERROR", b_X_Y_1000);
        }
    }
}

package binaryIV;

import gnu.trove.map.TIntIntMap;
import java.io.InputStream;
import java.util.concurrent.Callable;

public class BinaryIVAteSimulationTask implements Callable<String> {
    private final InputStream inputStream;

    public BinaryIVAteSimulationTask(InputStream inputStream) {
        this.inputStream = inputStream;
    }

    @Override
    public String call() {
        int Z = 0, X = 1, Y = 2, U = 3;

        try {
            TIntIntMap[] data = AteBounds.getDataFromCSV(inputStream, Z, X, Y);
            double[] bounds = AteBounds.getBoundsForBinaryIV(data, 100, 30, "ate", Z, X, Y, U);
            return String.format("%.4f,%.4f", bounds[0], bounds[1]);
        } catch (Exception e) {
            System.err.println("Simulation failed.");
            e.printStackTrace();
            return "ERROR,ERROR";
        }
    }
}

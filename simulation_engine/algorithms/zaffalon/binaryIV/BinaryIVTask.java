package binaryIV;

import gnu.trove.map.TIntIntMap;
import java.io.InputStream;
import java.util.concurrent.Callable;

public class BinaryIVTask {
    private final InputStream inputStream;
    private final String query;

    

    public BinaryIVTask(InputStream inputStream, String query) {
        this.inputStream = inputStream;
        this.query = query;
    }

    public String call() {
        int Z = 0, X = 1, Y = 2, U = 3;

        try {
            TIntIntMap[] data = BinaryIVBounds.getDataFromCSV(inputStream, Z, X, Y);
            double[] bounds = BinaryIVBounds.getBoundsForBinaryIV(data, 100, 30, query, Z, X, Y, U);
            return String.format("%.4f,%.4f", bounds[0], bounds[1]);
        } catch (Exception e) {
            System.err.println("Simulation failed.");
            e.printStackTrace();
            return "ERROR,ERROR";
        }
    }
}

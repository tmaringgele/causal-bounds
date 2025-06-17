package binaryIV;

import gnu.trove.map.TIntIntMap;
import java.io.InputStream;
import java.util.concurrent.Callable;

public class BinaryIVTask {

    //Compile:
    // javac -cp credici.jar binaryIV/*.java
    //Build JAR:
    // jar cf binaryIV/zaffalon.jar -C . binaryIV

    //Verify JAR:
    // jar tf binaryIV/zaffalon.jar

    //You should see something like:
    // binaryIV/BinaryIVAteSimulationTask.class
    // binaryIV/AteBounds.class


    private final InputStream inputStream;
    private final String query;
    private final boolean isConf;



    public BinaryIVTask(InputStream inputStream, String query, boolean isConf) {
        this.inputStream = inputStream;
        this.query = query;
        this.isConf = isConf;
    }

    public String call() {
        

        if (isConf) {
            int X = 0, Y = 1, U = 2;
            try {
                TIntIntMap[] data = BinaryConfBounds.getDataFromCSV(inputStream,  X, Y);
                double[] bounds = BinaryConfBounds.getBounds(data, 100, 30, query,  X, Y, U);
                return String.format("%.4f,%.4f", bounds[0], bounds[1]);
            } catch (Exception e) {
                System.err.println("Simulation failed.");
                e.printStackTrace();
                return "ERROR:"+e.getMessage();
            }
        } else {
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
}

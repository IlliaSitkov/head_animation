package org.example;

import java.io.*;
import java.util.*;
import org.opencv.core.*;

public class Model {
    private List<Point3> verts_;
    private List<int[]> faces_;

    public Model(String filename) {
        verts_ = new ArrayList<>();
        faces_ = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("v ")) {
                    String[] parts = line.split("\\s+");
                    Point3 v = new Point3(
                            Double.parseDouble(parts[1]),
                            Double.parseDouble(parts[2]),
                            Double.parseDouble(parts[3])
                    );
                    verts_.add(v);
                } else if (line.startsWith("f ")) {
                    String[] parts = line.split("\\s+");
                    int[] f = new int[parts.length - 1];
                    for (int i = 1; i < parts.length; i++) {
                        f[i - 1] = Integer.parseInt(parts[i].split("/")[0]) - 1;
                    }
                    faces_.add(f);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int nverts() {
        return verts_.size();
    }

    public int nfaces() {
        return faces_.size();
    }

    public Point3 vert(int i) {
        return verts_.get(i);
    }

    public int[] face(int idx) {
        return faces_.get(idx);
    }
}

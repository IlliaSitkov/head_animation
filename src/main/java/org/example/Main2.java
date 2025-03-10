package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

public class Main2 {
    private static float[][] zBuffer;
    private static final int WIDTH = 800;
    private static final int HEIGHT = 800;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Mat image = new Mat(HEIGHT, WIDTH, CvType.CV_8UC3, new Scalar(0, 0, 0));
        Model model = new Model("african_head.obj");

        int angle = 0;
        while (angle == 0) {
            image.setTo(new Scalar(0, 0, 0));
            drawTriangle(image, model, angle);
            System.out.println("ANGLE " + angle);
            HighGui.imshow("Display window", image);
            int key = HighGui.waitKey(10);
            if (key == 27) {  // Exit on ESC
                break;
            }
            angle = (angle + 5) % 360;
        }
        HighGui.destroyAllWindows();
    }

    private static void drawTriangle(Mat image, Model model, int angle) {
        zBuffer = new float[HEIGHT][WIDTH];
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                zBuffer[i][j] = Float.NEGATIVE_INFINITY;
            }
        }

        double[] lightDir = {0, 0, 1};

        for (int i = 0; i < model.nfaces(); i++) {
            int[] face = model.face(i);
            Point[] screenCoords = new Point[3];
            double[][] worldCoords = new double[3][3];
            double[] zPts = new double[3];

            for (int j = 0; j < 3; j++) {
                Point3 vertex = model.vert(face[j]);
                double[] v = rotateY(vertex, angle);
                screenCoords[j] = new Point((v[0] + 1) * WIDTH / 2, (1 - v[1]) * HEIGHT / 2);
                worldCoords[j] = v;
                zPts[j] = v[2];
            }

            double[] n = calculateNormal(worldCoords[0], worldCoords[1], worldCoords[2]);
            double intensity = dotProduct(n, lightDir);
            if (intensity > 0) {
                Scalar color = new Scalar(intensity * 255, intensity * 255, intensity * 255);
                triangle(screenCoords, zPts, image, color);
            }
        }
    }

    private static double[] rotateY(Point3 vertex, int angle) {
        double rad = Math.toRadians(angle);
        double[][] rotationMatrix = {
                {Math.cos(rad), 0, Math.sin(rad)},
                {0, 1, 0},
                {-Math.sin(rad), 0, Math.cos(rad)}
        };
        double[] coords = new double[]{vertex.x, vertex.y, vertex.z};
        return multiplyMatrixVector(rotationMatrix, coords);
    }

    private static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        double[] result = new double[3];
        for (int i = 0; i < 3; i++) {
            result[i] = matrix[i][0] * vector[0] + matrix[i][1] * vector[1] + matrix[i][2] * vector[2];
        }
        return result;
    }

    private static double[] calculateNormal(double[] v0, double[] v1, double[] v2) {
        double[] edge1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        double[] edge2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        double[] normal = crossProduct(edge1, edge2);
        double length = Math.sqrt(dotProduct(normal, normal));
        if (length == 0) return normal;
        for (int i = 0; i < 3; i++) normal[i] /= length;
        return normal;
    }

    private static double[] crossProduct(double[] a, double[] b) {
        return new double[]{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
        };
    }

    private static double dotProduct(double[] a, double[] b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    private static void triangle(Point[] pts, double[] zPts, Mat image, Scalar color) {
        Point bboxmin = new Point(WIDTH - 1, HEIGHT - 1);
        Point bboxmax = new Point(0, 0);
        Point clamp = new Point(WIDTH - 1, HEIGHT - 1);

        for (int i = 0; i < 3; i++) {
            bboxmin.x = Math.max(0, Math.min(bboxmin.x, pts[i].x));
            bboxmin.y = Math.max(0, Math.min(bboxmin.y, pts[i].y));
            bboxmax.x = Math.min(clamp.x, Math.max(bboxmax.x, pts[i].x));
            bboxmax.y = Math.min(clamp.y, Math.max(bboxmax.y, pts[i].y));
        }

        for (int x = (int) bboxmin.x; x <= bboxmax.x; x++) {
            for (int y = (int) bboxmin.y; y <= bboxmax.y; y++) {
                double[] bary = barycentric(pts, new Point(x, y));
                if (bary[0] < 0 || bary[1] < 0 || bary[2] < 0) continue;
                double z = bary[0] * zPts[0] + bary[1] * zPts[1] + bary[2] * zPts[2];
                if (z > zBuffer[y][x]) {
                    zBuffer[y][x] = (float) z;
                    image.put(y, x, color.val);
                }
            }
        }
    }

    private static double[] barycentric(Point[] pts, Point P) {
        double[] u = crossProduct(
                new double[]{pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x},
                new double[]{pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y}
        );
        if (Math.abs(u[2]) < 1) return new double[]{-1, 1, 1};
        return new double[]{1.0 - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]};
    }
}

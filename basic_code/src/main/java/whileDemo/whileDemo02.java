package whileDemo;

public class whileDemo02 {
    public static void main(String[] args) {
        //珠穆朗玛峰高度：8848.86米 = 8848860毫米
        //纸张厚度0.1毫米，对折多少次可以达到珠穆朗玛峰高度？
        double paper = 0.1;
        double height = 8848860;
        int count = 0;
        while (paper < height){
            paper *= 2;
            count++;
        }
        System.out.println("折叠"+count+"次");
    }
}

package method;

public class methodDemo03 {
    public static void main(String[] args) {
        //定义一个方法打印乘法表格

        printMullTable();
    }
    public static void printMullTable(){
        for (int i = 1; i <= 9; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(j + "*" + i + "=" + i * j + "\t");
            }
            System.out.println();

        }
    }

}

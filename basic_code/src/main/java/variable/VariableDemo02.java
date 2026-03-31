package variable;

import java.util.Scanner;

public class VariableDemo02 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入第一个数：");
        int a = sc.nextInt();
        System.out.println("请输入第二个数：");
        int b = sc.nextInt();


        // 判断是否为7的有缘数
/*        int ge = num % 10;
        int shi = num / 10 % 10;

        boolean result = ge == 7 || shi == 7 || num % 7 ==0;
        System.out.println(result);*/

        //三元运算符
        int max = a > b? a : b;
        System.out.println("最大值为：" + max);

    }
}

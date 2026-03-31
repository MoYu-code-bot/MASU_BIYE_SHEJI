package variable;

import java.util.Scanner;

public class VariableDemo01 {
    public static void main(String[] args) {
       /* Scanner sc = new Scanner(System.in);
        System.out.println("请输入一个三位数：");
        int num = sc.nextInt();
        boolean A = num % 3 ==0;
        System.out.println(A);*/
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入一个四位数：");
        int num = sc.nextInt();

        int ge = num % 10;
        int shi = num / 10 % 10;
        int bai = num / 100 % 10;
        int qian = num / 1000 % 10;
        System.out.println("个位：" + ge + "十位：" + shi + "百位：" + bai + "千位：" + qian);
        //判断是否是回文数
        boolean result = ge == qian && shi == bai;
        System.out.println("该四位数是回文数：" + result);


    }
}

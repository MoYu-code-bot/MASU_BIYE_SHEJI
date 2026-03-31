package ifDemo;

import java.util.Scanner;

public class ifDemo07 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入一个数字：");
        int num1 = sc.nextInt();
        System.out.println("请输入一个数字：");
        int num2 = sc.nextInt();

        int max = num1 > num2 ? num1 : num2;
        int min = num1 < num2 ? num1 : num2;

        int count = 0;
        for (int i = min; i <= max; i++) {
            if (i % 3 ==0 && i % 5 ==0){
                count++;
            }
        }
        if (count == 0){
            System.out.println("没有满足条件的数");
        }else {
            System.out.println("满足条件的数有：" + count + "个");
        }


    }
}

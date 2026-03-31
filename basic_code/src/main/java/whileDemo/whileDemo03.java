package whileDemo;

import java.util.Scanner;

public class whileDemo03 {
    public static void main(String[] args) {
        //数位之和：给定一个整数，请计算其所有数位之和。若n为负数，请先取其绝对值
        /*输入：12
        说明：1 + 2 = 3
        输出：3
        */
        Scanner sc = new Scanner(System.in);
        System.out.print("输入：");
        int n = sc.nextInt();
        if (n < 0){
            n = -n;
        }
        int sum = 0;
        while (n != 0){
            int temp = n % 10;
            sum += temp;
            n = n / 10;

        }
        System.out.println("输出：" + sum);


    }
}

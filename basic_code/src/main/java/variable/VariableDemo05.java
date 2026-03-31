package variable;

import java.util.Random;
import java.util.Scanner;

public class VariableDemo05 {
    public static void main(String[] args) {
        Random r = new Random();

        int n = r.nextInt(100) + 1;

        int countA = 0;
        int countB = 0;

        while (true) {
            Scanner sc = new Scanner(System.in);
            System.out.println("请输入数字：");
            int x = sc.nextInt();

            countA++;
            countB++;

            //触发大保底机制
            if (countB == 10){
                x = n;
                System.out.println("触发大保底机制，随机数字为：" + x);
            }
            if (x > n){
                System.out.println("数字太大");
            }else if(x < n){
                System.out.println("数字太小");
            }else {
                System.out.println("恭喜你猜对了");
                break;
            }
            //触发小保底机制
            if (countA % 3 == 0){
                System.out.println("触发小保底机制，随机数字的范围：" + (n-10) + "~" + (n+10));
            }
        }


    }
}

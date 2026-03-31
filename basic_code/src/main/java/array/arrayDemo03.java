package array;

import java.util.Scanner;

public class arrayDemo03 {
    public static void main(String[] args) {
        //动态初始化数组
       int[] arr = new int[5];
       Scanner sc = new Scanner(System.in);
        for (int i = 0; i < arr.length; i++) {
            System.out.println("请输入数字：");
            int num = sc.nextInt();
            arr[i] = num;
        }
        for (int i = 0; i < arr.length; i++) {
            System.out.println(arr[i]);
        }
    }
}

package array;

import java.util.Scanner;

public class arrayDemo04 {
    public static void main(String[] args) {
        //已知数组元素为{33,5,22,44,55,33}
        //键盘录入任意一个数据，查找数据是否存在数组中
        int[] arr = {33, 5, 22, 44, 55, 33};
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入一个数据：");
        int num = sc.nextInt();

        /*boolean flag = false;*/
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if(num == arr[i]){
                System.out.println(i);
                //flag = true;
                count++;
                break;
            }
        }
        if(count == 0){
            System.out.println("数据不存在");
        }

    }
}

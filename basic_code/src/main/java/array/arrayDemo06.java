package array;

import java.util.Random;
import java.util.Scanner;

public class arrayDemo06 {
    public static void main(String[] args) {
      //获取10个1-100之间的随机数并存入到数组当中，要求数组中不能有重复的元素
        int[] arr = new int[10];
        Random r = new Random();

        for (int i = 0; i < arr.length; ) {
            int num = r.nextInt(100) + 1;

            int count = 0;
            for (int j = 0; j < arr.length; j++) {
                if(num == arr[j]){
                    count++;
                    break;
                }
            }
            if (count == 0){
                arr[i] = num;
                i++;
            }
        }
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
    }
}

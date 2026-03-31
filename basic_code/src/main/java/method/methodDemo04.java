package method;

import java.util.Scanner;

public class methodDemo04 {
    public static void main(String[] args) {
      /*  作业 1：评委打分
          跳水比赛有五个评委打分，分数在 0~100 之间。
          最终得分会去掉一个最高分，去掉一个最低分，剩余的分数再求平均数，该平均数为选手最终得分。
          要求：
          1.利用键盘录入 5 个整数存入数组当中，如果分数超出范围需要重新录入
          2.定义方法分别求数组的最大值和最小值
          3.计算五名评委的总分
          4.总分 - 最大值 - 最小值，求选手最终平均分*/
        int[] arr = new int[5];

        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 5; i++) {
            System.out.println("请输入第" + (i + 1) + "个评委的打分：");
            int score = sc.nextInt();
            while (score < 0 || score > 100){
                System.out.println("输入的分数有误，请重新输入：");
                score = sc.nextInt();
            }
            arr[i] = score;
        }
        int max = getMax(arr);
        int min = getMin(arr);
        int sum = getSum(arr);

        System.out.println("最高分为：" + max);
        System.out.println("最低分为：" + min);
        System.out.println("总分为：" + sum);
        System.out.println("最终得分为：" + "("+sum + "-" + max + "-" + min + ")" + "/" + (arr.length - 2) +  "="+(sum - max - min) / (arr.length - 2));

    }
    public static int getMax(int[] arr){
        int max = arr[0];
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] > max){
                max = arr[i];
            }
        }
        return max;
    }
    public  static int getMin(int[] arr){
        int min =arr[0];
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] < min){
                min = arr[i];
            }
        }
        return min;
    }
    public static int getSum(int[] arr){
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum;

    }
}

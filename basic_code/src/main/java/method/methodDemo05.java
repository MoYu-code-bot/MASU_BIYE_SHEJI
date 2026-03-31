package method;

import java.util.Scanner;

public class methodDemo05 {
    public static void main(String[] args) {
       /* 计算班级分数
        班主任需要统计 10 名学生的数学成绩（0-100 分），计算及格率，平均分，并找出最高分。
        要求：
        1.键盘录入 10 名学生的成绩，存入数组。超出范围，提示 “成绩无效，请重新输入”。
        2.定义方法，求及格人数，根据及格人数，求及格率。
        3.定义方法求总分，根据总分求平均分
        4.定义方法求最大值。*/

        int[] arr = new int[10];
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 10; i++) {
            System.out.println("请输入第" + (i + 1) + "个学生的成绩：");
            int score = sc.nextInt();
            if (score < 0 || score > 100) {
                System.out.println("成绩输入错误，请重新输出：");
                score = sc.nextInt();
            }
            arr[i] = score;
        }
        int num = getCount(arr);
        System.out.println("及格人数为：" + num);
        int prob = num * 100 / arr.length;
        System.out.println("及格率为：" + prob + "%");
        int sum = getSum(arr);
        System.out.println("总分为：" + sum);
        int avg = sum / arr.length;
        System.out.println("平均分为：" + avg);
        int max = getMax(arr);
        System.out.println("最高分为：" + max);


    }
    //求及格人数
    public static int getCount(int[] arr){
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] >= 60){
                count++;
            }
        }
        return count;
    }

    //求总分
    public static int getSum(int[] arr){
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum;
    }
    //求最大值
    public static int getMax(int[] arr){
        int max = arr[0];
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] >  max){
                max = arr[i];
            }
        }
        return max;
    }
}

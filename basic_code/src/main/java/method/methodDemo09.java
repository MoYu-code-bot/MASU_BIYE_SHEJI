package method;

import java.util.Random;
import java.util.Scanner;

public class methodDemo09 {
    public static void main(String[] args) {
/*
        大乐透彩票号码生成
        题目规则
        彩票规则
        前区：1 ~ 35 选 5 个号码（号码唯一，不可重复）
        后区：1 ~ 12 选 2 个号码（号码唯一，不可重复）
        跨区号码可重复（前区和后区号码互不影响）
        中奖规则
        一等奖：5 + 2
        二等奖：5 + 1
        三等奖：5 + 0 / 4 + 2
        四等奖：4 + 1 / 3 + 2
        五等奖：4 + 0 / 3 + 1 / 2 + 2
        六等奖：3 + 0 / 1 + 2 / 2 + 1 / 0 + 2
        */
        //1.利用Random生成彩票号码


        int[] lotteryNumber = createLottery();
        printLotteryNumber(lotteryNumber);

        //2.利用Scanner模拟购买彩票
        int[] myLotteryNumber = buyLottery();
        printLotteryNumber(myLotteryNumber);
        System.out.println("前驱中了" + getCount(lotteryNumber, myLotteryNumber, 0, 4) + "个");
        System.out.println("后驱中了" + getCount(lotteryNumber, myLotteryNumber, 5, 6) + "个");

        //3.判断中奖情况
        getWinnerNumber(lotteryNumber, myLotteryNumber);


    }

    //判断arr2里面的数据, 在arr1中出现几次
    //arr1：中奖号码  arr2：用户购买的彩票号码
    public static void getWinnerNumber(int[] arr1, int[] arr2) {
        //判断前驱号码
        int count1 = getCount(arr1, arr2, 0, 4);
        //判断后驱代码
        int count2 = getCount(arr1, arr2, 5, 6);

        //判断中奖情况
        if (count1 == 5 && count2 == 2) {
            System.out.println("恭喜你，中了一等奖！");

        } else if (count1 == 5 && count2 == 1) {
            System.out.println("恭喜你，中二等奖！");
        } else if (count1 == 5 && count2 == 0 || count1 == 4 && count2 == 2) {
            System.out.println("恭喜你，中三等奖！");
        } else if (count1 == 4 && count2 == 1 || count1 == 3 && count2 == 2) {
            System.out.println("恭喜你，中四等奖！");
        } else if (count1 == 4 && count2 == 0 || count1 == 3 && count2 == 1 || count1 == 2 && count2 == 2) {
            System.out.println("恭喜你，中五等奖！");
        } else if (count1 == 3 && count2 == 0 || count1 == 2 && count2 == 1 || count1 == 1 && count2 == 2 || count1 == 0 && count2 == 2) {
            System.out.println("恭喜你，中六等奖！");
        } else {
            System.out.println("很遗憾，没有中奖~");
        }

    }

    public static int getCount(int[] arr1, int[] arr2, int start, int end) {
        int count = 0;
        for (int i = start; i <= end; i++) {
            boolean flag = contains(arr2[i], arr1, start, end);

            if (flag) {
                count++;
            }
        }
        return count;

    }

    public static int[] buyLottery() {
        int[] arr = new int[7];
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 5; ) {
            System.out.println("请输入第" + (i + 1) + "个彩票号码：");
            int number = sc.nextInt();
            if (number < 1 || number > 35) {
                System.out.println("输入的号码有误，请重新输入：");
                continue;

            }
            boolean flag = contains(number, arr, 0, 4);
            if (flag) {
                System.out.println("号码已存在，请重新选择：");
                continue;
            }
            arr[i] = number;
            i++;
        }
        for (int i = 0; i < 2; ) {
            System.out.println("请输入第" + (i + 6) + "个彩票号码：");
            int number = sc.nextInt();
            if (number < 1 || number > 12) {
                System.out.println("输入的号码有误，请重新输入：");
                continue;
            }
            boolean flag = contains(number, arr, 5, 6);
            if (flag) {
                System.out.println("号码已存在，请重新选择：");
                continue;
            }
            arr[i + 5] = number;
            i++;
        }
        return arr;
    }

    public static int[] createLottery() {
        int[] arr = new int[7];
        Random r = new Random();
        for (int i = 0; i < 5; ) {
            int number = r.nextInt(35) + 1;
            boolean flag = contains(number, arr, 0, 4);
            if (!flag) {
                arr[i] = number;
                i++;

            }
        }
        for (int i = 0; i < 2; ) {
            int number = r.nextInt(12) + 1;
            boolean flag = contains(number, arr, 5, 6);
            if (!flag) {
                arr[i + 5] = number;
                i++;
            }

        }


        return arr;
    }

    public static boolean contains(int number, int[] arr, int start, int end) {
        for (int i = start; i <= end; i++) {
            if (arr[i] == number) {
                return true;
            }
        }
        return false;
    }

    public static void printLotteryNumber(int[] arr) {
        System.out.print("中奖号码为：" + "[");
        for (int i = 0; i < arr.length; i++) {
            if (i == arr.length - 1) {

                System.out.print(arr[i]);
            } else {
                System.out.print(arr[i] + ", ");
            }
        }
        System.out.println(" ]");
    }
}

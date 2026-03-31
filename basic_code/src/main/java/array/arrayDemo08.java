package array;

public class arrayDemo08 {
    public static void main(String[] args) {
      /*  1、给定一个整数数组nums和一个整数目标值target,
      请你在该数组中找出和为目标值target的那两个整数，并输出他们的数组索引
      要求1：只要输出第一对满足要求的情况
      要求2：输出所有满足要求的情况
        */
        /*
        int[] arr = {10, 20, 30, 40, 50};

        int target = 50;
        //要求1：只要输出第一对满足要求
        for (int i = 0; i < arr.length; i++) {
            int n = arr[i];
            for (int j = i + 1; j < arr.length; j++) {
                int m = arr[j];
                int sum = n + m;
                if (sum == target) {
                    System.out.println("数组中两个数为：" + m + " " + n);
                    System.out.println("数组索引为：" + i + " " + j);
                }
            }
            break;
        }
        //要求2：输出所有满足要求的情况
        int m = 0;
        int n = 0;
        for (int i = 0; i < arr.length; i++) {
            n = arr[i];
            for (int j = 0; j < arr.length; j++) {
                m = arr[j];
                int sum = n + m;
                if (sum == target) {
                    System.out.println("数组中两个数为：" + m + " " + n);
                    System.out.println("数组索引为：" + i + " " + j);
                    break;
                }
            }
        }
        */



        /*2、给出两个有序数组arr1 和 arr2 ,将两个数组中的数据合并到一个大数组中

        */
        /*int[] arr1 = {1, 3, 5, 7, 9};
        int[] arr2 = {2, 4, 6, 8, 10};

        int sum = arr1.length + arr2.length;
        int[] arr3 = new int[sum];


        //合并
        //创建指针进行遍历
        int m = 0; //arr1的指针
        int n = 0; //arr2的指针
        int s = 0; //arr3的指针
        while (m < arr1.length && n < arr2.length){
            if(arr1[m] < arr2[n]){
                arr3[s] = arr1[m];
                m++;
            }else {
                arr3[s] = arr2[n];
                n++;
            }
            s++;
        }
        while (m < arr1.length){
            arr3[s] = arr1[m];
            m++;
            s++;
        }
        while (n < arr2.length){
            arr3[s] = arr2[n];
            n++;
            s++;
        }
        for (int j = 0; j < arr3.length; j++) {
            System.out.print(arr3[j] + " ");
        }
        System.out.println();
*/




        /*3、给定一个递增的有序数组和一个目标值，在数组中找到目标值，并打印其索引
        如果目标值不存在于数组中，打印应插入的位置,并将目标值插入数组中,打印数组信息
        */

        int[] arr = {1, 2, 4, 6, 8, 9, 10};
        int target = 5;

        int m = 0;
        for (int i = 0; i < arr.length; i++) {
            if(target == arr[i]){
                System.out.println("数组中索引为：" + i);
                break;

            }
            if(target < arr[i]){
                System.out.println("数组中插入的位置为：" + i);
                break;
            }
        }

/*        while (m < arr.length){
            if (target > arr[m]){
                m++;
            }else {
                target = arr[m];
                System.out.println(arr[m]);
                break;
            }
        }*/




    }
}

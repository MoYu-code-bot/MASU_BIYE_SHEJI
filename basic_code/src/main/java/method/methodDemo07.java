package method;

public class methodDemo07 {
    public static void main(String[] args) {
        //定义两个数组arr1和arr2，把两个数组元素合并到一个数组中，并按顺序进行输出
        //数组1；1,,3,5,7,9  数组2: 2,4
        int[] arr1 = {1, 3, 5, 7, 9};
        int[] arr2 = {2, 4};

        double result = getArr(arr1, arr2);
        System.out.println(result);
    }

    public static double getArr(int[] arr1, int[] arr2){

        int[] arr3 = new int[arr1.length + arr2.length];
        int index1 = 0;
        int index2 = 0;
        for (int i = 0; i < arr3.length; i++) {
            if(index1 == arr1.length){
                arr3[i] = arr2[index2];
                index2++;
                continue;
            }
            if(index2 == arr2.length){
                arr3[i] = arr1[index1];
                index1++;
                continue;
            }

            if(arr1[index1] < arr2[index2]){
                arr3[i] = arr1[index1];
                index1++;
            }else {
                arr3[i] = arr2[index2];
                index2++;
            }
        }

        if(arr3.length % 2 == 0){
            int num1 = arr3[arr3.length / 2];
            int num2 = arr3[arr3.length / 2 - 1];
            return (num1 + num2) / 2.0;

        }else {
            int num = arr3[arr3.length / 2];
            return num;

        }

    }

}

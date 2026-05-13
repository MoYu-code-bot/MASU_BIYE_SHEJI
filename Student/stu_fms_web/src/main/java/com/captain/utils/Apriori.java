package com.captain.utils;

import com.captain.entity.vo.AprioriElement;
import lombok.Data;

import java.util.*;

/**
 * Apriori先验算法工具
 * 这里有别于传统的Apriori算法，这里的每个频繁集必须至少包含一个就业单位
 * 这里的算法是把所有数据中同一家公司的学生个人成就进行分析，计算出频繁项
 */
public class Apriori {
    //频繁项的最小支持度
    private float minSupport=0.5F;

    //最小置信度
    private float minConfidence=0.75F;

    //保存寻找频繁集的过程中的记录集合
    private List<List<Long>> dataList;

    //原始数据
    private List<List<Long>> originalData;


    public static void main(String[] args) {
        List<List<Long>> data=new ArrayList<>();
        data.add(Arrays.asList(new Long[]{1L,2L}));
        data.add(Arrays.asList(new Long[]{2L,3L}));
        data.add(Arrays.asList(new Long[]{2L,3L,4L}));
        Apriori apriori = new Apriori(data);
        apriori.setMinSupport(0.5F);
        //开始计算频繁项
        Map<String, Object> map = apriori.computeFrequentItem();
    }

    public Apriori(List<List<Long>> originalData){
        this.originalData=originalData;
        convertData(originalData);
    }

    public float getMinSupport() {
        return minSupport;
    }

    public void setMinSupport(float minSupport) {
        this.minSupport = minSupport;
    }

    public float getMinConfidence() {
        return minConfidence;
    }

    public void setMinConfidence(float minConfidence) {
        this.minConfidence = minConfidence;
    }

    /**
     * 计算频繁项和对应的支持度
     * @return
     */
    public Map<String, Object>  computeFrequentItem(){
        List<List<Long>> resData=new ArrayList<>();
        List<Float> support=new ArrayList<>();
        Map<String, Object> map = supportFilter(dataList);
        dataList = (List<List<Long>>)map.get("data");
        resData.addAll(dataList);
        support.addAll((List<Float>)map.get("support"));
        int k=2;
        while(dataList.size()>0){
            //再次合并
            List<List<Long>> mergeData = merge(dataList, k++);
            //过滤支持度
            map = supportFilter(mergeData);
            //更新集合
            dataList = (List<List<Long>>)map.get("data");
            if(dataList.size()>0){
                //保存结果
                resData.addAll(dataList);
                support.addAll((List<Float>)map.get("support"));
            }
        }
//        //数据融合，如[2],[3],[2,3],融合成[2,3]
//        while(){
//
//        }
        Map<String, Object> resMap = new HashMap<>();
        resMap.put("data",resData);
        resMap.put("support",support);
        return resMap;
    }

    /**
     * 把数据转换为集合
     * 从:[[1,2],[3]]
     * 到：[[1],[2],[3]]
     * @param data
     */
    private void convertData(List<List<Long>> data){
        dataList=new ArrayList<>();
        Map<Long,Long> map=new HashMap<>();
        for(List<Long> list:data){
            for(Long item:list){
                if (map.containsKey(item)==false) {
                    map.put(item,item);
                    dataList.add(Arrays.asList(new Long[]{item}));
                }
            }
        }
    }

    /**
     * 获取两个列表中相同元素的个数
     * @param a
     * @param b
     * @return
     */
    private int getEqualCount(List<Long> a,List<Long> b){
        List<Long> objects = new ArrayList<>();
        objects.addAll(a);
        objects.retainAll(b);
        return objects.size();
    }

    /**
     * 取两个元素的去重并集
     * @param a
     * @param b
     * @return
     */
    private List<Long> unionListNoRepeat(List<Long> a,List<Long> b){
        List<Long> res=new ArrayList<>();
        //复制到res中
        res.addAll(a);
        //去重
        res.removeAll(b);
        //求并集
        res.addAll(b);
        return res;
    }
//
//    public static void main(String[] args) {
//        List<Integer> list1=new ArrayList<>();
//        List<Integer> list2=new ArrayList<>();
//        list1.addAll(Arrays.asList(new Integer[]{1,2}));
//        list2.addAll(Arrays.asList(new Integer[]{2,3}));
//        System.out.println(list1);
//        list1.removeAll(list2);
//        list1.addAll(list2);
//        System.out.println(list1);
//    }

    /**
     * 把集合中的元素合并成为一个新的集合
     * 如：[[1],[2],[3],[A],[B]]
     * 合并成： [[1,2],[1,3]....]
     * @param data
     * @param k 需要合并成为几个元素在一起的集合
     * @return
     */
    private List<List<Long>> merge(List<List<Long>> data,int k){
        List<List<Long>> result=new ArrayList<>();
        for(int i=0;i<data.size();i++){
            for(int j=i+1;j<data.size();j++){
                //两个list必须有k-2个元素相同，才能被合并
                if(getEqualCount(data.get(i),data.get(j))==k-2){
                    result.add(unionListNoRepeat(data.get(i),data.get(j)));
                }
            }
        }
        //这里还可以优化，可以去除掉重复的集合
        //
        //
        return result;
    }

    /**
     * 计算支持度
     * 支持度就是：在原始的每条记录中包含这个集合的记录条数的比例
     * @param list
     * @return
     */
    private float computeSupport(List<Long> list){
        //包含的记录条数
        int containCount=0;
        for(List<Long> item:originalData){
            if(item.size()>=list.size()){
                //要满足子集，那么原来的集合大小必须要大于等于这个集合
                List<Long> temp = new ArrayList<>();
                temp.addAll(item);
                //取交集
                temp.retainAll(list);
                if(temp.size()==list.size()){
                    containCount++;
                }
            }
        }
        float support=(float)(containCount*1.0/originalData.size());
        if((int)support==1){
            return 0.999999F;
        }else{
            return support;
        }
    }

    /**
     * 支持度过滤，剔除小于最小支持度的集合
     * @param data 合并出来的集合
     * @return
     */
    private Map<String,Object> supportFilter(List<List<Long>> data){
        List<List<Long>> newData=new ArrayList<>();
        List<Float> supportList=new ArrayList<>();
        for(List<Long> item:data){
            float support=computeSupport(item);
            if (support>=minSupport) {
                //满足最小支持度才加入进去
                newData.add(item);
                supportList.add(support);
            }
        }
        //返回结果
        Map<String, Object> map = new HashMap<>();
        map.put("data",newData);
        map.put("support",supportList);
        return map;
    }

}

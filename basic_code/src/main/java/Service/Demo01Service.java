package Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Demo01Service {


}
   /*
   //用到任何外部数据之前，先判空
   if (ObjectUtil.isEmpty(xxx)){
        return 默认值;
     }

    //BeanUtils.copyProperties批量拷贝
    TargetObj target = new TargetObj();
    BeanUtils.copyProperties(source, target);


    //Stream提取字段  从对象列表中提取某个字段组成新列表
    List<String> codeList = list.stream()
    .map(XXX::getCode)
    .collect(Collectors.toList());

    //批量查询 + Map  循环前批量查好，循环内直接取
    Map<String, Integer> map = service.batchQuery(codeList);
    map.getOrDfault(key, 默认值);

    //JSON解析  数据库存的是 JSON 字符串，但需要返回数组
    if (ObjectUtil.isNotNull(jsonStr)) {
        List<String> list = JSON.parseArray(jsonStr, String.class);
    }

    //遍历 + 转换 + 收集  把 A 类型列表转成 B 类型列表
    List<B> resultList = new ArrayList<>();
    aList.forEach(a -> {
        B b = new B();
        BeanUtils.copyProperties(a, b);
        //处理特殊字段....
        resultList.add(b);
    })
   */


    /*
   public PageResult<List<StrategyInfoResp>> getStrategyList(
   ){

   }
    */









   /* public PageResult<List<响应类>> getXxxList(筛选参数, 分页参数) {

        // ① 调接口拿数据
        PageResult<List<实体类>> pageResult = client.getXxxList(...);
        List<响应类> respList = new ArrayList<>();

        // ② 防御判断
        if (pageResult.isSuccess()) {
            if (ObjectUtil.isEmpty(pageResult.getData())) {
                return PageResult.empty();
            }

            // ③ 循环前：批量查需要的关联数据
            List<String> codeList = pageResult.getData().stream()
                    .map(实体类::getCode).collect(Collectors.toList());
            Map<String, XXX> xxxMap = client.batchQueryXxx(codeList);

            // ④ 循环：逐条转换
            pageResult.getData().forEach(entity -> {
                响应类 resp = new 响应类();
                BeanUtils.copyProperties(entity, resp);
                // 处理特殊字段（JSON解析、改名、额外查询等）
                resp.setXxx(xxxMap.getOrDefault(entity.getCode(), 默认值));
                respList.add(resp);
            });
        }

        // ⑤ 包装返回
        return PageResult.success(respList, pageResult.getPagination());
    }
    */

    /*
    //搭建外框架 -> 调用接口查询数据 -> 设置转换器 -> 判空 -> 批量查询数据 -> 循环体 -> 处理特殊字段 -> 返回封装结果
    public PageResult<List<响应类>> getXxxx(){

    PageResult<List<实体类>> 方法名 = **Client.getXxxx(

    );
    List<响应类> 方法名1 = new ArrayList<>();//创建空列表，不局限于创建空列表




    }




    */
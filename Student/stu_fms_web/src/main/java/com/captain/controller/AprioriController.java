package com.captain.controller;

import com.alibaba.fastjson.JSON;
import com.captain.common.Result;
import com.captain.utils.Apriori;
import lombok.Data;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * Apriori算法
 */
@RestController
public class AprioriController {

    /**
     * 计算支持度的频繁项
     * @return
     */
//    @GetMapping("/apriori")
//    public Result getSupport(Float minSupport,List<List<Long>> data){
//        Apriori apriori = new Apriori(data);
//        if(minSupport!=null){
//            apriori.setMinSupport(minSupport);
//        }
//        return Result.success(apriori.computeFrequentItem());
//    }

    @PostMapping("/apriori")
    public Result getSupport(@RequestBody AprioriEntity entity){
        Apriori apriori = new Apriori(entity.getData());
        Float minSupport=entity.getMinSupport();
        if(minSupport!=null){
            apriori.setMinSupport(minSupport);
        }
        return Result.success(apriori.computeFrequentItem());
    }
}

@Data
class  AprioriEntity{
    Float minSupport;
    List<List<Long>> data;
}

package com.captain.controller;

import com.captain.common.Result;
import com.captain.entity.po.Dictionary;
import com.captain.service.DictionaryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/dict")
public class DictionaryController {
    @Autowired
    DictionaryService dictionaryService;

    @GetMapping("/all")
    public Result getAll(){
        List<Dictionary> list = dictionaryService.list();
        return Result.success(list);
    }

    /**
     * 根据key来获取对应的数据
     * @param key
     * @return
     */
    @GetMapping("/listByKey")
    public Result getListByKey(String key){
        List<Dictionary> dictionaries = dictionaryService.listByKey(key);
        return Result.success(dictionaries);
    }

    /**
     * 根据key和code获取唯一的一个字典状态
     * @param key
     * @param code
     * @return
     */
    @GetMapping("/getOneByKeyAndCode")
    public Result getOne(String key,Long code){
        Dictionary oneByKeyAndCode = dictionaryService.getOneByKeyAndCode(key, code);
        return Result.success(oneByKeyAndCode);
    }

}

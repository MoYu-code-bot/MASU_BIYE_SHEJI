package com.captain.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.entity.po.Dictionary;
import com.captain.mapper.DictionaryMapper;
import com.captain.service.DictionaryService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Service
public class DictionaryServiceImpl extends ServiceImpl<DictionaryMapper, Dictionary> implements DictionaryService {

    @Autowired
    private DictionaryMapper dictionaryMapper;

    @Override
    public Long getDictCodeBydictValue(String name) {
        return dictionaryMapper.getDictCodeBydictValue(name);
    }

    @Override
    public List<Dictionary> listByKey(String key) {
        return baseMapper.selectList(new QueryWrapper<Dictionary>()
        .eq("dict_key",key));
    }

    @Override
    public Dictionary getOneByKeyAndCode(String key, Long code) {
        return baseMapper.selectOne(new QueryWrapper<Dictionary>()
                .eq("dict_key",key)
                .eq("dict_code",code));
    }
}

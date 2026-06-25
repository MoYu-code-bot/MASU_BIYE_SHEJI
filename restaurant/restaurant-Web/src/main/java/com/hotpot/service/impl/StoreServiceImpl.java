package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.PageResult;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Store;
import com.hotpot.mapper.StoreMapper;
import com.hotpot.service.StoreService;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;

@Service
public class StoreServiceImpl extends ServiceImpl<StoreMapper, Store> implements StoreService {

    @Override
    public PageResult<Store> pageQuery(PageQuery query) {
        Page<Store> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<Store> wrapper = new LambdaQueryWrapper<>();
        if (StringUtils.hasText(query.getKeyword())) {
            wrapper.like(Store::getName, query.getKeyword())
                    .or().like(Store::getAddress, query.getKeyword());
        }
        wrapper.orderByDesc(Store::getCreateTime);
        return PageResult.of(page(page, wrapper));
    }

    @Override
    public List<Store> listAll() {
        return list(new LambdaQueryWrapper<Store>()
                .eq(Store::getStatus, 1)
                .orderByAsc(Store::getId));
    }
}

package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.PageResult;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Dish;
import com.hotpot.mapper.DishMapper;
import com.hotpot.service.DishService;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;

@Service
public class DishServiceImpl extends ServiceImpl<DishMapper, Dish> implements DishService {

    @Override
    public PageResult<Dish> pageQuery(PageQuery query, Long categoryId, Long storeId, Integer status) {
        Page<Dish> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<Dish> wrapper = new LambdaQueryWrapper<>();
        if (categoryId != null) {
            wrapper.eq(Dish::getCategoryId, categoryId);
        }
        if (storeId != null) {
            wrapper.and(w -> w.eq(Dish::getStoreId, storeId).or().isNull(Dish::getStoreId));
        }
        if (status != null) {
            wrapper.eq(Dish::getIsOnSale, status);
        }
        if (StringUtils.hasText(query.getKeyword())) {
            wrapper.like(Dish::getName, query.getKeyword());
        }
        wrapper.orderByAsc(Dish::getSortOrder).orderByDesc(Dish::getCreateTime);
        return PageResult.of(page(page, wrapper));
    }

    @Override
    public List<Dish> listByStoreId(Long storeId) {
        return list(new LambdaQueryWrapper<Dish>()
                .eq(Dish::getStoreId, storeId)
                .or()
                .isNull(Dish::getStoreId)
                .eq(Dish::getIsOnSale, 1)
                .orderByAsc(Dish::getSortOrder));
    }

    @Override
    public List<Dish> listByCategoryId(Long categoryId) {
        return list(new LambdaQueryWrapper<Dish>()
                .eq(Dish::getCategoryId, categoryId)
                .eq(Dish::getIsOnSale, 1)
                .orderByAsc(Dish::getSortOrder));
    }
}

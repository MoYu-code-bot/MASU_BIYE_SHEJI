package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.entity.Category;
import com.hotpot.mapper.CategoryMapper;
import com.hotpot.service.CategoryService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CategoryServiceImpl extends ServiceImpl<CategoryMapper, Category> implements CategoryService {

    @Override
    public List<Category> listByStoreId(Long storeId) {
        LambdaQueryWrapper<Category> wrapper = new LambdaQueryWrapper<>();
        if (storeId != null) {
            wrapper.and(w -> w.eq(Category::getStoreId, storeId).or().isNull(Category::getStoreId));
        }
        wrapper.orderByAsc(Category::getSortOrder);
        return list(wrapper);
    }

    @Override
    public List<Category> listAll() {
        return list(new LambdaQueryWrapper<Category>().orderByAsc(Category::getSortOrder));
    }
}

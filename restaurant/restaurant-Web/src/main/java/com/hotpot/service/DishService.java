package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.common.PageResult;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Dish;

import java.util.List;

public interface DishService extends IService<Dish> {

    PageResult<Dish> pageQuery(PageQuery query, Long categoryId, Long storeId, Integer status);

    List<Dish> listByStoreId(Long storeId);

    List<Dish> listByCategoryId(Long categoryId);
}

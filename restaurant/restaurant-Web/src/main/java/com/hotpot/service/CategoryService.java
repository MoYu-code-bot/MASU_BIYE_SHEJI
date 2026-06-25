package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.Category;

import java.util.List;

public interface CategoryService extends IService<Category> {

    List<Category> listByStoreId(Long storeId);

    List<Category> listAll();
}

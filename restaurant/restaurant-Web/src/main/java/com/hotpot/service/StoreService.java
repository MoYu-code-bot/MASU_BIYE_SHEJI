package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.common.PageResult;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Store;

import java.util.List;

public interface StoreService extends IService<Store> {

    PageResult<Store> pageQuery(PageQuery query);

    List<Store> listAll();
}

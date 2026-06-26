package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Category;
import com.hotpot.service.CategoryService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "C端-分类接口")
@RestController
@RequestMapping("/api/categories")
@RequiredArgsConstructor
public class CategoryApiController {

    private final CategoryService categoryService;

    @GetMapping("list")
    @ApiOperation("获取分类列表")
    public Result<List<Category>> list(@ApiParam("门店ID") @RequestParam(required = false) Long storeId) {
        return Result.success(categoryService.listByStoreId(storeId));
    }
}

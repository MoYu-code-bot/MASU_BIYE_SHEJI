package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Category;
import com.hotpot.service.CategoryService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "C端-分类接口")
@RestController
@RequestMapping("/api/categories")
@RequiredArgsConstructor
public class CategoryApiController {

    private final CategoryService categoryService;

    @ApiOperation("获取分类列表")
    @GetMapping
    public Result<List<Category>> list() {
        return Result.success(categoryService.listByStoreId(null));
    }
}

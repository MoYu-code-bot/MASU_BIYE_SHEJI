package com.hotpot.controller.admin;

import com.hotpot.common.Result;
import com.hotpot.entity.Category;
import com.hotpot.service.CategoryService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Api(tags = "B端-分类管理")
@RestController
@RequestMapping("/admin/categories")
@RequiredArgsConstructor
public class AdminCategoryController {

    private final CategoryService categoryService;

    @ApiOperation("查询全部分类")
    @GetMapping
    public Result<List<Category>> list() {
        return Result.success(categoryService.listAll());
    }

    @ApiOperation("新增分类")
    @PostMapping
    public Result<?> add(@ApiParam("分类信息") @RequestBody Category category) {
        categoryService.save(category);
        return Result.success();
    }

    @ApiOperation("修改分类")
    @PutMapping("/update")
    public Result<?> update(@ApiParam("分类ID") @RequestParam Long categoryId,
                            @ApiParam("分类信息") @RequestBody Category category) {
        category.setId(categoryId);
        categoryService.updateById(category);
        return Result.success();
    }

    @ApiOperation("删除分类")
    @DeleteMapping("/delete")
    public Result<?> delete(@ApiParam("分类ID") @RequestParam Long categoryId) {
        categoryService.removeById(categoryId);
        return Result.success();
    }
}

class Explanation {
    constructor(class_names) {
        this.names = class_names;
        if (class_names.length < 10) {
            this.colors = d3.scale.category10().domain(this.names);
            this.colors_i = d3.scale.category10().domain(d3.range(this.names.length));
        } else {
            this.colors = d3.scale.category20().domain(this.names);
            this.colors_i = d3.scale.category20().domain(d3.range(this.names.length));
        }
    }

    show(exp, label, div) {
        var svg = div.append('svg').style('width', '100%');
        var colors = ['#5F9EA0', this.colors_i(label)];
        var names = ['NOT ' + this.names[label], this.names[label]];
        if (this.names.length == 2) {
            colors = [this.colors_i(0), this.colors_i(1)];
            names = this.names;
        }
        var plot = new Barchart(svg, exp, true, names, colors, true, 10);
        svg.style('height', plot.svg_height + 'px');
    }

    show_raw_text(exp, label, raw, div, opacity = true) {
        var colors = ['#5F9EA0', this.colors_i(label)];
        if (this.names.length == 2) {
            colors = [this.colors_i(0), this.colors_i(1)];
        }
        var word_lists = [[], []];
        var max_weight = -1;
        for (let [word, start, weight] of exp) {
            if (weight > 0) {
                word_lists[1].push([start, start + word.length, weight]);
            } else {
                word_lists[0].push([start, start + word.length, -weight]);
            }
            max_weight = Math.max(max_weight, Math.abs(weight));
        }
        if (!opacity) {
            max_weight = 0;
        }
        this.display_raw_text(div, raw, word_lists, colors, max_weight, true);
    }

    show_raw_tabular(exp, label, div) {
        div.classed('lime', true).classed('table_div', true);
        var colors = ['#5F9EA0', this.colors_i(label)];
        if (this.names.length == 2) {
            colors = [this.colors_i(0), this.colors_i(1)];
        }
        var table = div.append('table');
        var thead = table.append('tr');
        thead.append('td').text('Feature');
        thead.append('td').text('Value');
        thead.style('color', 'black').style('font-size', '20px');
        for (let [fname, value, weight] of exp) {
            var tr = table.append('tr');
            tr.style('border-style', 'hidden');
            tr.append('td').text(fname);
            tr.append('td').text(value);
            if (weight > 0) {
                tr.style('background-color', colors[1]);
            } else if (weight < 0) {
                tr.style('background-color', colors[0]);
            } else {
                tr.style('color', 'black');
            }
        }
    }

    hexToRgb(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    applyAlpha(hex, alpha) {
        var components = this.hexToRgb(hex);
        return `rgba(${components.r},${components.g},${components.b},${alpha.toFixed(3)})`;
    }

    display_raw_text(div, raw_text, word_lists = [], colors = [], max_weight = 1, positions = false) {
        div.classed('lime', true).classed('text_div', true);
        div.append('h3').text('Text with highlighted words');
        var highlight_tag = 'span';
        var text_span = div.append('span').style('white-space', 'pre-wrap').text(raw_text);
        var position_lists = word_lists;
        if (!positions) {
            position_lists = this.wordlists_to_positions(word_lists, raw_text);
        }
        var objects = [];
        for (let i of d3.range(position_lists.length)) {
            position_lists[i].map(x => objects.push({ 'label': i, 'start': x[0], 'end': x[1], 'alpha': max_weight === 0 ? 1 : x[2] / max_weight }));
        }
        objects = _.sortBy(objects, x => x['start']);
        var node = text_span.node().childNodes[0];
        var subtract = 0;
        for (let obj of objects) {
            var word = raw_text.slice(obj.start, obj.end);
            var start = obj.start - subtract;
            var end = obj.end - subtract;
            var match = document.createElement(highlight_tag);
            match.appendChild(document.createTextNode(word));
            match.style.backgroundColor = this.applyAlpha(colors[obj.label], obj.alpha);
            var after = node.splitText(start);
            after.nodeValue = after.nodeValue.substring(word.length);
            node.parentNode.insertBefore(match, after);
            subtract += end;
            node = after;
        }
    }

    wordlists_to_positions(word_lists, raw_text) {
        var ret = [];
        for (let words of word_lists) {
            if (words.length === 0) {
                ret.push([]);
                continue;
            }
            var re = new RegExp("\\b(" + words.join('|') + ")\\b", 'gm');
            var temp;
            var list = [];
            while ((temp = re.exec(raw_text)) !== null) {
                list.push([temp.index, temp.index + temp[0].length]);
            }
            ret.push(list);
        }
        return ret;
    }
}

class Barchart {
    constructor(svg, data, horizontal, names, colors, show_legend, margin_top) {
        this.svg = svg;
        this.data = data;
        this.horizontal = horizontal;
        this.names = names;
        this.colors = colors;
        this.show_legend = show_legend;
        this.margin_top = margin_top;
        this.draw();
    }

    draw() {
        var margin = { top: this.margin_top, right: 20, bottom: 30, left: 40 };
        var width = +this.svg.style('width').replace('px', '') - margin.left - margin.right;
        var height = 300 - margin.top - margin.bottom;

        var x = d3.scale.linear().range([0, width]);
        var y = d3.scale.ordinal().rangeRoundBands([0, height], .1);

        var g = this.svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        x.domain(d3.extent(this.data, function (d) { return d[1]; })).nice();
        y.domain(this.data.map(function (d) { return d[0]; }));

        g.append('g')
            .attr('class', 'x axis')
            .attr('transform', 'translate(0,' + height + ')')
            .call(d3.svg.axis().scale(x).orient('bottom').ticks(10, '%'));

        g.append('g')
            .attr('class', 'y axis')
            .call(d3.svg.axis().scale(y).orient('left'));

        g.selectAll('.bar')
            .data(this.data)
            .enter().append('rect')
            .attr('class', 'bar')
            .attr('x', function (d) { return x(Math.min(0, d[1])); })
            .attr('y', function (d) { return y(d[0]); })
            .attr('width', function (d) { return Math.abs(x(d[1]) - x(0)); })
            .attr('height', y.rangeBand());

        if (this.show_legend) {
            var legend = this.svg.selectAll('.legend')
                .data(this.colors.slice().reverse())
                .enter().append('g')
                .attr('class', 'legend')
                .attr('transform', function (d, i) { return 'translate(0,' + i * 20 + ')'; });

            legend.append('rect')
                .attr('x', width - 18)
                .attr('width', 18)
                .attr('height', 18)
                .style('fill', function (d, i) { return d; });

            legend.append('text')
                .attr('x', width - 24)
                .attr('y', 9)
                .attr('dy', '.35em')
                .style('text-anchor', 'end')
                .text(function (d, i) { return this.names[i]; }.bind(this));
        }
        this.svg_height = height + margin.top + margin.bottom;
    }
}

export class Matrix {
  private _matrix: number[][];
  private _rowsCount: number;
  private _columnsCount: number;

  constructor(array: number[][]) {
    this._matrix = array;
    this._rowsCount = array.length;
    this._columnsCount = array[0].length;
    this.validate();
  }

  private validate() {
    for (const row of this._matrix) {
      if (row.length !== this._columnsCount) {
        throw new Error("Invalid matrix");
      }
    }
  }

  public get rowsCount() {
    return this._rowsCount;
  }

  public get columnsCount() {
    return this._columnsCount;
  }

  public get matrix() {
    return this._matrix;
  }
}
